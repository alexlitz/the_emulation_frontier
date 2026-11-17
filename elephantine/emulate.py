"""Write a checkpoint in a HOST architecture whose weights make the host compute a TARGET model.

  python -m elephantine.emulate TARGET HOST --out DIR
  python -m elephantine.emulate HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M --out smol-in-360m
  python -m elephantine.emulate TARGET HOST --plan-only          # the plan and whether the script builds it

It downloads the TARGET's weights and only the HOST's config.json (the host's own weights are never
used) and writes DIR in the host's real architecture: the host's own config.json and stock
`transformers` class (loadable with AutoModelForCausalLM, no remote code), with weights that make
it compute the target. Then it verifies: the target and the written checkpoint on a few prompts in
fp32 (and fp64 with the models' internal fp32 casts lifted, when it fits in memory), max |delta
logits| and greedy-continuation agreement, and writes README.md and plan.json into DIR.
DIR must be new, empty or an earlier output of this script (it carries the .tef-emulate-output marker,
and only then is it deleted and rewritten); any other directory is refused.

The pair must be an edge of the headline relation (graph.model_graph.emulates). The plan printed is
graph.edge_plan.edge_plan, built from graph.reduce's own pieces; a pair that is not an edge, an edge
that needs a limit / regime / floor the script does not build, and an exact edge outside the script's
scope are refused with that plan and a non-zero exit (2, 3, 4). Nothing is approximated silently: the
only non-exact constructions are the limits the relation itself labels (LIMIT_BUILT: ReLU <- GeLU),
labelled LIMIT in the notes, README.md and plan.json.

What the host may change (the post's rules): any weight, plus the RoPE theta / scaling (the
inv_freq), the norm eps, the attention window sizes, and the vocab size. The relation ignores the
vocab (the embedding and LM head are a resizable front end), so the host's vocab_size is set to the
target's, its bos / eos / pad ids to the target's, and -- when the target's head is untied and the
host ties it -- tie_word_embeddings to false. Every such change is listed in plan.json / README.md.

Scope (FAMILIES), every family through its stock transformers class only:
  RMSNorm decoders   llama, mistral, qwen2, qwen3, granite, gemma, gemma2, gemma3_text, phi3, glm, ernie4_5,
                     seed_oss, helium, apertus (xIELU), lfm2 (gated short-conv layers), olmo2 / olmo3
                     (post-norm only, whole-projection qk-norm), granitemoehybrid (Granite-4.0 / -4.0-H: RoPE or
                     NoPE, Mamba-2 layers), bamba (Mamba-2 hybrid), deepseek_v3 (MLA), as targets and hosts;
                     falcon_h1 as a host (its Mamba-2 branch zeroed);
  LayerNorm decoders gpt_neox (Pythia; parallel or sequential residual, partial RoPE), stablelm, falcon
                     (RoPE; multi-query / grouped / per-head qkv, one or two norms), gpt2, gpt_neo, gpt_bigcode
                     and opt (learned absolute positions), olmo (no norm gains: they go into the readers), as
                     targets and hosts; cohere as a host (one shared norm, parallel block);
  MoE                mixtral, qwen2_moe, qwen3_moe, ernie4_5_moe, granitemoe, glm4_moe, phimoe, olmoe,
                     flex_olmo, deepseek_v3, granitemoehybrid (tiny / small) as hosts for a dense target and as
                     targets in an MoE host whose routing reproduces theirs;
  recurrent          mamba (Mamba-1) in a bigger mamba; the Mamba-2 layers of bamba / granitemoehybrid targets.
Refused outright, with the reason (REFUSED_FAMILIES): MLA deepseek_v2, dots1's unflagged qk-norm, jamba's
NoPE attention, gpt_oss's sinks and clamped activation.
The constructions, each exact:
  hidden   the residual stream x (d dims) is carried as U x (D dims). RMSNorm host, RMSNorm target:
           U = a [I; 0] (zero-padded); RMSNorm over D dims with the tail at 0 is sqrt(d/D) times the
           norm over d dims, so eps' = a^2 eps d/D and every gain read by a projection is scaled by
           sqrt(d/D) (elephantine/hidden_width.py). LayerNorm host, LayerNorm target: U = a [I; 1 1^T/d]
           (the padded dims carry the mean, so the host's mean is the target's and the centred tail is
           0). LayerNorm host, RMSNorm target: U = a [I; -I; 0] (the [h; -h] mirror, mean 0, D >= 2d).
           A host whose eps is fixed (OLMo, OPT) takes a = sqrt(eps_h D / (eps_t n d)) instead of eps'.
           Every writer (embedding, positions, o_proj, down_proj and their biases) is post-composed
           with U; readers read the first d dims. A qk-norm shares the eps: q and k are scaled so that
           their rms over the (padded) head is the same factor times the target's.
  layers   the target's layers are spread over the host's; every other layer is the identity
           (its residual writers zeroed; hf/full_emulation.py). Under ZERO_EDGES a full-window target
           layer only goes on a host full-window layer. A sequential target layer on a parallel-block
           host (GPT-NeoX / StableLM / Falcon parallel residual, Cohere) takes two host layers, attention
           then FFN, each with the other branch zeroed (seq <= parallel, block/parallel_residual.py). A
           parallel target needs a parallel host (with one shared norm only if the target has one).
  heads    extra query heads zeroed; each target KV head is tiled over as many host KV groups as its
           query heads need (repeat_interleave, hf/gqa_mha.py), or -- KV-head PACKING (graph/model_graph.py
           kv_hosts_needed) -- several target KV heads sit side by side in one wider host head, each query reading
           its own block. head_dim zero-padded. A host whose rotary reads a longrope config
           (ROPE_PAIRS_FREE_TYPES) takes a per-pair table (short = long factor, theta 1: the target's fp32
           inverse frequencies bit for bit): each packed block's pairs at the target's frequencies, the target's
           unrotated (or NoPE) dims on pairs at frequency 0. Else RoPE: target pair
           i (rotate_half or interleaved, either layout) goes on host pair k i when the host rotates
           k times as many dims (the same frequency at the same theta), else on pair i with
           theta' = theta^(rot_h / rot_t) (hf/partial_rope_host.py); unrotated dims on unrotated dims.
           The host's inv_freq is recomputed from its written config and checked against the target's.
  FFN      zero-padded units. A dense target in an MoE host: every expert holds the FFN with the router
           at 0 and the gates renormalised (or rescaled by E/k when they are not), the FFN in the shared
           expert (a sigmoid gate at 0 -> x 2), or -- ERNIE-4.5, whose selection bias picks the experts
           -- dealt over the k selected experts, each gate 1/k. An MoE target in an MoE host: the router
           rows copied, each target expert duplicated m = k_h / k_t times (softmax over the copies sums
           back to the target's gate, the top-k_h picks every copy), unused experts never selected (ERNIE's
           selection bias) with renormalised gates, the shared expert copied (hf/moe_dense.py).
  windows  a sliding-window target layer gets the host's window set to the target's (hf/local_full.py); the
           window lever: a full target layer on a host with a window gets it removed (or its layer type set to
           full); a sliding target on a host with no window setting is the relation's mask edge (noted: the
           window is the caller's attention mask, needed only past the window).
  routers  an RMSNorm MoE host with a free hidden dim but no selection bias gets a constant residual channel
           (its square taken out of the eps): router rows on it select the experts that hold a dense FFN (+R)
           or keep spare experts out (-R). Expert copies may each hold a slice (m e_h >= e_t, down x m); sigmoid
           routers, selection biases, routed_scaling_factor, norm_topk_prob and groups as the code reads them.
  norms    post-norm-only blocks (OLMo-2 / 3): the branch-output norms' gains x a sqrt(d / D); a whole-projection
           qk-norm holds r = target / host entries: q, k x gam / sqrt(r), gains x sqrt(r).
  MLA      latents held m times (the hard-coded latent-norm eps sees one copy's rms), heads / nope / v padded.
  mixers   Mamba-1: channels, state, dt rank padded, conv taps right-aligned. Mamba-2 (full-width gated norm, y
           without a free scale): every target channel held m times, eps kept, the stream carried x sqrt(D / d).
           LFM2 short conv: channels padded, taps right-aligned. Hybrid hosts: attention on attention layers,
           mixers on mixers, in order; the rest zeroed.
  scales   attention scale, embedding / residual / logit / muP multipliers (Granite, Gemma, Cohere,
           Falcon-H1) are folded into the projections and norm gains.
  positions learned absolute tables (GPT-2, OPT) are copied row for row and lifted like every writer.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch                                                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

F64 = torch.float64
NOT_AN_EDGE, APPROX_EDGE, OUT_OF_SCOPE = 2, 3, 4
PROMPTS = ["The capital of France is", "def fibonacci(n):", "Once upon a time, there was a little"]
NEW_TOKENS = 16
GITHUB = "https://github.com/alexlitz/the_emulation_frontier"


class Refused(Exception):
    def __init__(self, code, reasons):
        self.code = code
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        super().__init__("; ".join(self.reasons))


# --- families -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fam:
    name: str
    norm: str = "rms"            # "rms" | "ln" (LayerNorm, weight and bias) | "ln_w" (weight only: Cohere) |
    #                              "ln_none" (no weight or bias: OLMo; OPT without elementwise_affine)
    norm_offset: float = 0.0     # Gemma RMSNorm multiplies by (1 + w)
    sandwich: bool = False       # Gemma-2 / -3: post-attention and post-FFN norms on the branch outputs
    post_only: bool = False      # OLMo-2 / 3, FlexOlmo: the post-norms only (x + norm(f(x)), no pre-norm)
    mla: bool = False            # DeepSeek-V3 multi-head latent attention
    qk_kind: str = "head"        # the qk-norm variant: "head" (per head, one head_dim gain) | "proj" (one RMSNorm over
    #                              the whole q / k projection: OLMo-2 / 3, OLMoE, FlexOlmo)
    layout: str = "llama"        # weight names (see _spec): "llama" | "neox" | "gpt2" (Conv1D) | "opt" | "falcon"
    fused_attn: bool = False     # Phi-3: one qkv_proj
    fused_mlp: str = ""          # "" | module name of the fused [gate; up] projection (Phi-3, GLM, Granite-4.0)
    rope: str = "half"           # rotary pairs: "half" (i, i + rot/2) | "inter" (2i, 2i + 1) | "none" (learned positions)
    block: str = "seq"           # "seq" | "neox" (config.use_parallel_residual, two norms) | "shared" (parallel, one
    #                              norm: Cohere) | "shared_cfg" (config.use_parallel_residual, one norm: StableLM) |
    #                              "falcon" (parallel_attn / new_decoder_architecture, num_ln_in_parallel_attn norms)
    ffn: str = "glu"             # "glu" (gate, up, down) | "mlp" (up, act, down)
    window: str = "none"         # "none": no window setting; "global": one sliding_window for every layer;
    #                              "layers": per-layer layer_types + one sliding_window; "flag": use_sliding_window
    #                              turns one sliding_window on (Qwen2/3-MoE: the script only switches it off)
    qk_norm: object = False      # per-head q_norm / k_norm modules (RMSNorm): True / False / a config key
    moe: str = ""                # "", "mixtral" (block_sparse_moe, w1 / w3 / w2), "qwen" (mlp.experts / shared),
    #                              "ernie" (mlp.experts / shared_experts, selection bias moe_statics),
    #                              "granite" (block_sparse_moe: stacked input_linear / output_linear, router.layer)
    act_key: str = "hidden_act"
    bias: tuple = ("attention_bias", "attention_bias")   # (q/k/v bias, o bias): True / False / a config key
    mlp_bias: object = False     # True / False / a config key
    eps_key: str = "rms_norm_eps"
    eps_fixed: float = 0.0       # the norm eps when the config has none (OLMo, OPT: 1e-5)
    theta_key: str = "rope_theta"
    part_key: str = "partial_rotary_factor"
    mult: str = ""               # multipliers: "gemma" (embedding sqrt(d)), "granite" (embedding / residual / attention /
    #                              logits), "cohere" (logit_scale), "mup" (Falcon-H1, and its Mamba-2 branch, zeroed)
    names: tuple = ()            # module-name overrides for _spec, as (key, value) pairs
    host_only: str = ""          # why the family is not read as a target


GRANITE = dict(mult="granite", mlp_bias="mlp_bias")
FAMILIES = {f.name: f for f in [
    Fam("llama", mlp_bias="mlp_bias"),
    Fam("mistral", window="global", bias=(False, False)),
    Fam("qwen2", window="layers", bias=(True, False)),
    Fam("qwen3", window="layers", qk_norm=True),
    Fam("granite", **GRANITE),
    Fam("gemma", norm_offset=1.0, mult="gemma"),
    Fam("gemma2", norm_offset=1.0, sandwich=True, window="layers", act_key="hidden_activation", mult="gemma"),
    Fam("gemma3_text", norm_offset=1.0, sandwich=True, window="layers", qk_norm=True, act_key="hidden_activation",
        mult="gemma"),
    Fam("phi3", fused_attn=True, fused_mlp="gate_up_proj", window="global", bias=(False, False)),
    Fam("glm", rope="inter", fused_mlp="gate_up_proj", bias=("attention_bias", False)),
    Fam("seed_oss", bias=("attention_bias", "attention_out_bias"), mlp_bias="mlp_bias"),
    Fam("ernie4_5", rope="inter", bias=("use_bias", "use_bias"), mlp_bias="use_bias"),
    Fam("olmo", norm="ln_none", eps_key="", eps_fixed=1e-5),
    Fam("stablelm", norm="ln", block="shared_cfg", bias=("use_qkv_bias", False), eps_key="layer_norm_eps"),
    Fam("granitemoehybrid", fused_mlp="input_linear", names=(("mlp", "shared_mlp"), ("down", "output_linear")),
        **GRANITE),
    Fam("olmo2", post_only=True, qk_norm=True, qk_kind="proj"),
    Fam("olmo3", post_only=True, qk_norm=True, qk_kind="proj", window="layers"),
    Fam("olmoe", qk_norm=True, qk_kind="proj", moe="olmoe"),
    Fam("flex_olmo", post_only=True, qk_norm=True, qk_kind="proj", moe="olmoe"),
    Fam("mamba", eps_key="layer_norm_epsilon", rope="none", layout="mamba"),
    Fam("apertus", ffn="mlp", qk_norm=True, names=(("ln_in", "attention_layernorm"), ("ln_ffn", "feedforward_layernorm"))),
    Fam("lfm2", eps_key="norm_eps", qk_norm=True, act_key="",
        names=(("o", "out_proj"), ("ln_in", "operator_norm"), ("ln_ffn", "ffn_norm"), ("mlp", "feed_forward"),
               ("gate", "w1"), ("up", "w3"), ("down", "w2"), ("final", "model.embedding_norm"), ("qn", "q_layernorm"),
               ("kn", "k_layernorm"), ("mixer", "conv")), bias=(False, False)),
    Fam("deepseek_v3", mla=True, moe="glm4", bias=("attention_bias", "attention_bias")),
    Fam("helium", rope="inter", bias=("attention_bias", False), mlp_bias="mlp_bias"),
    Fam("gpt_bigcode", norm="ln", layout="bigcode", rope="none", ffn="mlp", bias=(True, True), mlp_bias=True,
        eps_key="layer_norm_epsilon", act_key="activation_function"),
    Fam("gpt_neo", norm="ln", layout="gptneo", rope="none", ffn="mlp", window="neo", bias=(False, True), mlp_bias=True,
        eps_key="layer_norm_epsilon", act_key="activation_function"),
    Fam("bamba", names=(("mlp", "feed_forward"), ("ln_ffn", "pre_ff_layernorm"), ("final", "model.final_layernorm"))),
    Fam("mixtral", window="global", moe="mixtral", bias=(False, False)),
    Fam("phimoe", norm="ln", window="global", moe="phimoe", bias=("attention_bias", "attention_bias")),
    # Qwen2-MoE / Qwen3-MoE slide when use_sliding_window is set (sliding_window is None otherwise): Qwen3-MoE on
    # every layer; Qwen2-MoE on every layer under the eager / sdpa mask but only on layers >= max_window_layers under
    # flash-attention. The script sets no window on them (a sliding target is the mask edge); it switches the
    # host's own off (verify/REVIEW.md 5)
    Fam("qwen2_moe", window="flag", moe="qwen", bias=(True, False)),
    Fam("qwen3_moe", window="flag", moe="qwen", qk_norm=True),
    Fam("ernie4_5_moe", rope="inter", moe="ernie", bias=("use_bias", "use_bias"), mlp_bias="use_bias"),
    Fam("glm4_moe", moe="glm4", qk_norm="use_qk_norm", bias=("attention_bias", False)),
    Fam("granitemoe", moe="granite", **GRANITE),
    Fam("gpt_neox", norm="ln", layout="neox", block="neox", ffn="mlp", mlp_bias=True, eps_key="layer_norm_eps",
        theta_key="rotary_emb_base", part_key="rotary_pct"),
    Fam("gpt2", norm="ln", layout="gpt2", rope="none", ffn="mlp", bias=(True, True), mlp_bias=True,
        eps_key="layer_norm_epsilon", act_key="activation_function"),
    Fam("opt", norm="ln", layout="opt", rope="none", ffn="mlp", bias=("enable_bias", "enable_bias"),
        mlp_bias="enable_bias", eps_key="", eps_fixed=1e-5, act_key="activation_function"),
    Fam("falcon", norm="ln", layout="falcon", block="falcon", ffn="mlp", bias=("bias", "bias"), mlp_bias="bias",
        eps_key="layer_norm_epsilon", act_key="activation"),
    Fam("cohere", norm="ln_w", rope="inter", block="shared", eps_key="layer_norm_eps", mult="cohere",
        host_only="cohere: a parallel-block LayerNorm target is not read (hosts only)"),
    Fam("falcon_h1", mult="mup", mlp_bias="mlp_bias",
        names=(("mlp", "feed_forward"), ("ln_ffn", "pre_ff_layernorm"), ("final", "model.final_layernorm")),
        host_only="falcon_h1: a hybrid (Mamba-2) target; the script reads softmax-attention targets"),
]}
TARGET_FAMILIES = {k for k, f in FAMILIES.items() if not f.host_only}

# Stock families the script refuses outright, and why (a construction the relation counts but the host's stock
# class cannot carry exactly, or one not built)
REFUSED_FAMILIES = {
    "deepseek_v2": "MLA attention (the latent RMSNorm and the shared decoupled RoPE key)",
    "dots1": "per-head qk-norm (an RMSNorm on q and k) the relation's classify does not flag: a target without "
             "one is the qk-norm regime edge, not exact",
    "jamba": "attention without positional encoding (NoPE): a RoPE target needs rotations",
    "gpt_oss": "attention sinks and the clamped swigluoai activation (exact only in the clamp regime)",
}


def _src(c, s):
    """A bias / flag source: True, False, or a config attribute."""
    return s if isinstance(s, bool) else bool(getattr(c, s, False))


def _spec(fam, c=None):
    """Module names of a family: the state-dict keys of layer j are f"{layers}.{j}.{attn}.{q}.weight" etc."""
    if fam.layout == "neox":
        s = dict(layers="gpt_neox.layers", attn="attention", qkv="query_key_value", o="dense",
                 ln_in="input_layernorm", ln_ffn="post_attention_layernorm", mlp="mlp", up="dense_h_to_4h",
                 down="dense_4h_to_h", embed="gpt_neox.embed_in", pos=None, final="gpt_neox.final_layer_norm",
                 lm="embed_out")
    elif fam.layout == "gpt2":
        s = dict(layers="transformer.h", attn="attn", qkv="c_attn", o="c_proj", ln_in="ln_1", ln_ffn="ln_2",
                 mlp="mlp", up="c_fc", down="c_proj", embed="transformer.wte", pos="transformer.wpe",
                 final="transformer.ln_f", lm="lm_head")
    elif fam.layout == "falcon":
        s = dict(layers="transformer.h", attn="self_attention", qkv="query_key_value", o="dense",
                 ln_in="input_layernorm", ln_ffn="post_attention_layernorm", mlp="mlp", up="dense_h_to_4h",
                 down="dense_4h_to_h", embed="transformer.word_embeddings", pos=None, final="transformer.ln_f",
                 lm="lm_head")
        if c is not None and _parallel(c, fam) and not _one_norm(c, fam):
            s.update(ln_in="ln_attn", ln_ffn="ln_mlp")
    elif fam.layout == "bigcode":
        s = dict(layers="transformer.h", attn="attn", qkv="c_attn", o="c_proj", ln_in="ln_1", ln_ffn="ln_2",
                 mlp="mlp", up="c_fc", down="c_proj", embed="transformer.wte", pos="transformer.wpe",
                 final="transformer.ln_f", lm="lm_head")
    elif fam.layout == "gptneo":
        s = dict(layers="transformer.h", attn="attn.attention", q="q_proj", k="k_proj", v="v_proj", o="out_proj",
                 ln_in="ln_1", ln_ffn="ln_2", mlp="mlp", up="c_fc", down="c_proj", embed="transformer.wte",
                 pos="transformer.wpe", final="transformer.ln_f", lm="lm_head")
    elif fam.layout == "opt":
        s = dict(layers="model.decoder.layers", attn="self_attn", q="q_proj", k="k_proj", v="v_proj", o="out_proj",
                 ln_in="self_attn_layer_norm", ln_ffn="final_layer_norm", mlp="", up="fc1", down="fc2",
                 embed="model.decoder.embed_tokens", pos="model.decoder.embed_positions",
                 final="model.decoder.final_layer_norm", lm="lm_head")
    else:
        s = dict(layers="model.layers", attn="self_attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj",
                 ln_in="input_layernorm", ln_ffn="post_attention_layernorm", mlp="mlp", gate="gate_proj",
                 up="up_proj", down="down_proj", embed="model.embed_tokens", pos=None, final="model.norm",
                 lm="lm_head")
        if fam.fused_attn:
            s["qkv"] = "qkv_proj"
        if fam.fused_mlp:
            s["gate_up"] = fam.fused_mlp
        if fam.sandwich:
            s.update(ln_ffn="pre_feedforward_layernorm", post_attn="post_attention_layernorm",
                     post_ffn="post_feedforward_layernorm")
        if fam.post_only:
            s.update(ln_in=None, ln_ffn=None, post_attn="post_attention_layernorm",
                     post_ffn="post_feedforward_layernorm")
    s.setdefault("qn", "q_norm")
    s.setdefault("kn", "k_norm")
    s.setdefault("mixer", "mamba")
    s.update(dict(fam.names))
    if c is not None and _one_norm(c, fam) and not fam.post_only:
        s["ln_ffn"] = None
    return s


def _key(*parts):
    return ".".join(p for p in parts if p)


POS_OFFSET = {"opt": 2}         # the row of position 0 in a family's learned position table

# activation names with the same mathematics
ACT_MATH = {"silu": "silu", "swish": "silu", "gelu": "gelu_erf", "gelu_python": "gelu_erf",
            "gelu_new": "gelu_tanh", "gelu_pytorch_tanh": "gelu_tanh", "gelu_fast": "gelu_tanh",
            "gelu_accurate": "gelu_tanh", "relu": "relu", "relu2": "relu2", "quick_gelu": "quick_gelu",
            "mish": "mish", "tanh": "tanh", "sigmoid": "sigmoid"}
# RoPE types whose inv_freq is recomputed from the sequence length at run time: a head_dim change cannot keep
# them (the dynamic rescale depends on the rotated dims)
RUNTIME_ROPE = {"dynamic"}


# --- configs ------------------------------------------------------------------------------------------

def fetch_config(repo):
    """The raw config.json dict of a Hub repo or a local directory (only the config is downloaded)."""
    p = Path(repo)
    if p.is_dir():
        return json.loads((p / "config.json").read_text())
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo, "config.json")).read_text())


_RESOLVED = {}


def resolved(raw):
    """The transformers config object for a raw dict (defaults filled in, layer_types computed)."""
    hit = _RESOLVED.get(id(raw))
    if hit is not None and hit[0] is raw:
        return hit[1]
    from transformers import CONFIG_MAPPING
    mt = raw.get("model_type")
    if mt not in CONFIG_MAPPING:
        raise Refused(OUT_OF_SCOPE, f"model_type {mt!r} has no stock transformers config")
    d = {k: v for k, v in raw.items() if k not in ("auto_map", "quantization_config")}
    with warnings_off():
        c = CONFIG_MAPPING[mt].from_dict(d)
    _RESOLVED[id(raw)] = (raw, c)
    return c


@contextlib.contextmanager
def warnings_off():
    import logging
    import warnings
    lg = logging.getLogger("transformers")
    old = lg.level
    lg.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            lg.setLevel(old)


# remote-code class names whose model is the one transformers ported natively (same weights, same computation):
# Phi-3.5-MoE's config names its remote PhiMoEForCausalLM, the native port is PhimoeForCausalLM. The written config
# names the stock class.
STOCK_ALIASES = {"PhiMoEForCausalLM": "PhimoeForCausalLM"}


def family_of(raw, role):
    """The family of a raw config, or a refusal. role: 'target' or 'host'."""
    mt = raw.get("model_type")
    allowed = TARGET_FAMILIES if role == "target" else set(FAMILIES)
    if mt not in allowed:
        raise Refused(OUT_OF_SCOPE, f"{role}: {FAMILIES[mt].host_only}" if mt in FAMILIES else
                      f"{role}: {mt} -- {REFUSED_FAMILIES[mt]}" if mt in REFUSED_FAMILIES else
                      f"{role}: model_type {mt!r} is not a supported family")
    arch = (raw.get("architectures") or [None])[0]
    arch = STOCK_ALIASES.get(arch, arch)
    if arch is not None:
        import transformers
        cls = getattr(transformers, arch, None)
        if cls is None or getattr(getattr(cls, "config_class", None), "model_type", None) != mt:
            raise Refused(OUT_OF_SCOPE, f"{role}: architecture {arch} is not the stock transformers class "
                                        f"of {mt!r} (remote code)")
    return FAMILIES[mt]


HD_FROM_HIDDEN = {"gpt_bigcode", "gpt_neo", "olmoe", "phimoe", "stablelm", "granitemoehybrid", "granitemoe", "gpt_neox", "gpt2", "opt", "falcon"}   # ignore config.head_dim


def _head_dim(c):
    if c.model_type in HD_FROM_HIDDEN:
        return c.hidden_size // c.num_attention_heads
    return getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads


def _n_kv(c):
    if c.model_type == "falcon":
        return c.num_kv_heads if c.new_decoder_architecture else 1 if c.multi_query else c.num_attention_heads
    if c.model_type == "gpt_bigcode":
        return 1 if c.multi_query else c.num_attention_heads
    return getattr(c, "num_key_value_heads", None) or c.num_attention_heads


def _inter(c, fam):
    if fam.name == "gpt2":
        return c.n_inner or 4 * c.hidden_size
    if fam.name == "gpt_neo":
        return c.intermediate_size or 4 * c.hidden_size
    if fam.name == "gpt_bigcode":
        return c.n_inner or 4 * c.hidden_size
    if fam.name == "lfm2":
        w = c.intermediate_size
        if c.block_auto_adjust_ff_dim:
            w = int(2 * w / 3)
            if c.block_ffn_dim_multiplier is not None:
                w = int(c.block_ffn_dim_multiplier * w)
                w = c.block_multiple_of * ((w + c.block_multiple_of - 1) // c.block_multiple_of)
        return w
    if fam.name == "opt":
        return c.ffn_dim
    if fam.name == "falcon":
        return c.ffn_hidden_size
    if fam.name == "granitemoehybrid":
        return c.shared_intermediate_size
    return c.intermediate_size


def _attn_layers(c, fam):
    """Per layer: does the host layer have a softmax attention block (hybrids: Bamba, Granite-4.0-H)?"""
    n = c.num_hidden_layers
    if fam.name == "bamba":
        return [t == "attention" for t in c.layers_block_type]
    if fam.name == "lfm2":
        return [t == "full_attention" for t in c.layer_types]
    if fam.name == "granitemoehybrid":
        return [t == "attention" for t in (getattr(c, "layer_types", None) or ["attention"] * n)]
    return [True] * n


def _m2_dims(c):
    I = int(c.mamba_expand * c.hidden_size)
    return {"I": I, "H": c.mamba_n_heads, "P": c.mamba_d_head, "G": c.mamba_n_groups, "N": c.mamba_d_state,
            "K": c.mamba_d_conv, "conv_bias": bool(c.mamba_conv_bias), "proj_bias": bool(c.mamba_proj_bias)}


def _m2_fit(tc, tfam, hc, hfam):
    """The Mamba-2 copy construction (graph/model_graph.py RC_M2) for a hybrid target's Mamba-2 layers, or why not.
    The stock gated RMSNorm (Bamba, Granite-4.0-H) spans every channel and y = C h + D x has no free scale (x, B, C
    are silu outputs), so the host holds each target channel m times: I_h = m I_t, each target head's channels as m
    consecutive copies, every host head inside one target head's block (P_h divides m P_t), every host B / C group
    inside one target group's span; state, conv zero-padded / right-aligned."""
    if hfam.name not in ("bamba", "granitemoehybrid"):
        return f"mixer: the target's Mamba-2 layers need a host with Mamba-2 layers (host family {hfam.name})"
    t, h = _m2_dims(tc), _m2_dims(hc)
    if h["I"] % t["I"]:
        return (f"Mamba-2: host inner width {h['I']} is not a multiple of the target's {t['I']} (the gated norm "
                f"spans every channel: copies keep its rms, zero channels do not)")
    m = h["I"] // t["I"]
    if (m * t["P"]) % h["P"]:
        return f"Mamba-2: host head_dim {h['P']} does not divide {m} x the target's {t['P']}"
    if (m * t["I"] // t["G"]) % (h["I"] // h["G"]):
        return (f"Mamba-2: a host B / C group ({h['I'] // h['G']} channels) straddles the target's groups "
                f"({m} x {t['I'] // t['G']})")
    for k in ("N", "K"):
        if h[k] < t[k]:
            return f"Mamba-2: host {'state' if k == 'N' else 'conv'} {h[k]} < target {t[k]}"
    if t["conv_bias"] and not h["conv_bias"]:
        return "Mamba-2: the target's conv has a bias, the host's has none"
    if t["proj_bias"] and not h["proj_bias"]:
        return "Mamba-2: the target's in / out projections have biases, the host's have none"
    return {"t": t, "h": h, "m": m}


def _mla_dims(c):
    return {"H": c.num_attention_heads, "ql": c.q_lora_rank or 0, "kl": c.kv_lora_rank, "nope": c.qk_nope_head_dim,
            "rope": c.qk_rope_head_dim, "v": c.v_head_dim, "inter": bool(getattr(c, "rope_interleave", True))}


def _mla_plan(tc, tfam, hc, hfam, hraw, changes, notes, reasons):
    """MLA <- MLA (graph/model_graph.py: latents an integer multiple, the latent norms' eps hard-coded): each latent
    held m times (the rms of m copies is the rms of one), kv_b / q_b read the first copy; heads, the nope and v head
    dims zero-padded; the shared rotary key's pairs placed as dim_map places a head's. Returns the RoPE note."""
    if not (tfam.mla and hfam.mla):
        reasons.append("attention: MLA on one side only (the latent RMSNorm and the shared rotary key have no exact "
                       "construction in another attention)")
        return ""
    t, h = _mla_dims(tc), _mla_dims(hc)
    if h["kl"] % t["kl"]:
        reasons.append(f"MLA: host kv_lora_rank {h['kl']} is not a multiple of the target's {t['kl']} (the latent norm's "
                       f"eps is hard-coded: copies keep its rms, zero dims do not)")
    if bool(t["ql"]) != bool(h["ql"]):
        reasons.append("MLA: one side has a q latent (q_lora_rank), the other a plain q_proj")
    elif t["ql"] and h["ql"] % t["ql"]:
        reasons.append(f"MLA: host q_lora_rank {h['ql']} is not a multiple of the target's {t['ql']}")
    for k in ("H", "nope", "v"):
        if h[k] < t[k]:
            reasons.append(f"MLA: host {k} {h[k]} < target {t[k]}")
    try:
        dm, hpair, power = dim_map(t["rope"], t["rope"], "inter" if t["inter"] else "half", h["rope"], h["rope"],
                                   "inter" if h["inter"] else "half")
    except Refused as e:
        reasons.append(e.reasons[0])
        return ""
    r = _rope_plan(tc, tfam, hc, hfam, hraw, t["rope"], h["rope"], hpair, power, changes, notes)
    if isinstance(r, list):
        reasons += r
        return ""
    _MLA_PLAN[id(changes)] = {"t": t, "h": h, "dm": dm}
    return r


_MLA_PLAN = {}


def _write_mla(put, pre, T, M, d, D, wf, lift, alpha, shapes):
    """A target MLA block (state-dict tensors T, or None: an identity layer) into the host's at prefix pre."""
    t, h, dm = M["t"], M["h"], torch.tensor(M["dm"])
    z = lambda *sh: torch.zeros(*sh, dtype=F64)
    Ht, qkt, qkh = t["H"], t["nope"] + t["rope"], h["nope"] + h["rope"]
    qrow = torch.cat([torch.arange(t["nope"]), h["nope"] + dm])          # target q dim -> host q dim, per head
    rows = torch.cat([hh * qkh + qrow for hh in range(Ht)])               # target q rows -> host q rows
    out = {}
    if t["ql"]:
        mq = h["ql"] // t["ql"]
        A, Ab, g, B = z(h["ql"], D), z(h["ql"]), z(h["ql"]), z(h["H"] * qkh, h["ql"])
        if T is not None:
            for c in range(mq):
                A[c * t["ql"]:(c + 1) * t["ql"], :d] = T["q_a_proj.weight"]
                g[c * t["ql"]:(c + 1) * t["ql"]] = T["q_a_layernorm.weight"]
                if "q_a_proj.bias" in T:
                    Ab[c * t["ql"]:(c + 1) * t["ql"]] = T["q_a_proj.bias"]
            B[rows, :t["ql"]] = T["q_b_proj.weight"] * alpha
        out.update({"q_a_proj.weight": A, "q_a_proj.bias": Ab, "q_a_layernorm.weight": g, "q_b_proj.weight": B})
    else:
        Q = z(h["H"] * qkh, D)
        if T is not None:
            Q[rows, :d] = T["q_proj.weight"] * alpha
        out["q_proj.weight"] = Q
    mk = h["kl"] // t["kl"]
    KA, KAb, kg = z(h["kl"] + h["rope"], D), z(h["kl"] + h["rope"]), z(h["kl"])
    KB = z(h["H"] * (h["nope"] + h["v"]), h["kl"])
    O, Ob = z(D, h["H"] * h["v"]), z(D)
    if T is not None:
        ka = T["kv_a_proj_with_mqa.weight"]
        for c in range(mk):
            KA[c * t["kl"]:(c + 1) * t["kl"], :d] = ka[:t["kl"]]
            kg[c * t["kl"]:(c + 1) * t["kl"]] = T["kv_a_layernorm.weight"]
            if "kv_a_proj_with_mqa.bias" in T:
                KAb[c * t["kl"]:(c + 1) * t["kl"]] = T["kv_a_proj_with_mqa.bias"][:t["kl"]]
        KA[h["kl"] + dm, :d] = ka[t["kl"]:]
        if "kv_a_proj_with_mqa.bias" in T:
            KAb[h["kl"] + dm] = T["kv_a_proj_with_mqa.bias"][t["kl"]:]
        kb = T["kv_b_proj.weight"].view(Ht, t["nope"] + t["v"], t["kl"])
        KBv = KB.view(h["H"], h["nope"] + h["v"], h["kl"])
        KBv[:Ht, :t["nope"], :t["kl"]] = kb[:, :t["nope"]]
        KBv[:Ht, h["nope"]:h["nope"] + t["v"], :t["kl"]] = kb[:, t["nope"]:]
        ov = O.view(D, h["H"], h["v"])
        ov[:, :Ht, :t["v"]] = lift(T["o_proj.weight"].T, wf).T.reshape(D, Ht, t["v"])
        if "o_proj.bias" in T:
            Ob = lift(T["o_proj.bias"], wf)
    out.update({"kv_a_proj_with_mqa.weight": KA, "kv_a_proj_with_mqa.bias": KAb, "kv_a_layernorm.weight": kg,
                "kv_b_proj.weight": KB, "o_proj.weight": O, "o_proj.bias": Ob})
    for k, v in out.items():
        if pre + k in shapes:
            put(pre + k, v)
        elif k.endswith(".bias") and v.abs().max().item() != 0:
            raise Refused(OUT_OF_SCOPE, f"bias: the target's MLA {k} is nonzero; the host has none")


def _conv_fit(tc, tfam, hc, hfam):
    """LFM2's gated short conv (y = C * conv(B * x) over the hidden channels) in a wider one (graph/model_graph.py
    RC_GE): zero channels give B x = 0 and C = 0; the kernel's taps right-aligned."""
    if hfam.name != "lfm2":
        return f"mixer: the target's short-conv layers need a host with them (host family {hfam.name})"
    if hc.conv_L_cache < tc.conv_L_cache:
        return f"conv: host kernel {hc.conv_L_cache} < target {tc.conv_L_cache}"
    if tc.conv_bias and not hc.conv_bias:
        return "conv: the target's short conv has biases, the host's has none"
    return {"Kt": tc.conv_L_cache, "Kh": hc.conv_L_cache, "bias": bool(hc.conv_bias)}


def _write_shortconv(put, pre, T, M, d, D, wf, lift):
    z = lambda *sh: torch.zeros(*sh, dtype=F64)
    W = z(3 * D, D)
    Wt = T["in_proj.weight"]
    for blk in range(3):                                         # [B | C | x]
        W[blk * D:blk * D + d, :d] = Wt[blk * d:(blk + 1) * d]
    put(pre + "in_proj.weight", W)
    C = z(D, 1, M["Kh"])
    C[:d, :, M["Kh"] - M["Kt"]:] = T["conv.weight"]
    put(pre + "conv.weight", C)
    O = z(D, D)
    O[:, :d] = lift(T["out_proj.weight"].T, wf).T
    put(pre + "out_proj.weight", O)
    if M["bias"]:
        b = z(3 * D)
        if "in_proj.bias" in T:
            for blk in range(3):
                b[blk * D:blk * D + d] = T["in_proj.bias"][blk * d:(blk + 1) * d]
        put(pre + "in_proj.bias", b)
        cb = z(D)
        if "conv.bias" in T:
            cb[:d] = T["conv.bias"]
        put(pre + "conv.bias", cb)
        put(pre + "out_proj.bias", lift(T["out_proj.bias"], wf) if "out_proj.bias" in T else z(D))


def _write_m2(put, pre, T, M, d, D, wf, a, lift):
    """A target Mamba-2 mixer (state-dict tensors T) into a host mixer at prefix pre (the M2 layout of _m2_fit)."""
    t, h, m = M["t"], M["h"], M["m"]
    It, Ih, Pt, Ph, Ht, Hh = t["I"], h["I"], t["P"], h["P"], t["H"], h["H"]
    Gt, Gh, Nt, Nh, Kt, Kh = t["G"], h["G"], t["N"], h["N"], t["K"], h["K"]
    blk = m * Pt
    ch = torch.arange(Ih)
    src = (ch // blk) * Pt + (ch % blk) % Pt                     # host channel -> target channel
    first = (ch % blk) < Pt                                       # the copy out_proj reads
    head_src = (torch.arange(Hh) * Ph) // blk                    # host head -> target head
    gw_h = Ih // Gh
    grp_src = ((torch.arange(Gh) * gw_h) // (m * It // Gt))      # host group -> target group
    z = lambda *sh: torch.zeros(*sh, dtype=F64)
    Wt, bt = T["in_proj.weight"], T.get("in_proj.bias")
    # target in_proj rows: [gate It | x It | B Gt Nt | C Gt Nt | dt Ht]
    t_off = {"g": 0, "x": It, "B": 2 * It, "C": 2 * It + Gt * Nt, "dt": 2 * It + 2 * Gt * Nt}
    h_off = {"g": 0, "x": Ih, "B": 2 * Ih, "C": 2 * Ih + Gh * Nh, "dt": 2 * Ih + 2 * Gh * Nh}
    rows = torch.full((2 * Ih + 2 * Gh * Nh + Hh,), -1, dtype=torch.long)
    rows[h_off["g"]:h_off["g"] + Ih] = t_off["g"] + src
    rows[h_off["x"]:h_off["x"] + Ih] = t_off["x"] + src
    for gh in range(Gh):
        gt = int(grp_src[gh])
        if gt < Gt:
            for key in ("B", "C"):
                rows[h_off[key] + gh * Nh:h_off[key] + gh * Nh + Nt] = t_off[key] + gt * Nt + torch.arange(Nt)
    rows[h_off["dt"]:h_off["dt"] + Hh] = t_off["dt"] + head_src
    used = rows >= 0
    W = z(rows.numel(), D)
    W[used, :d] = Wt[rows[used]]
    put(pre + "in_proj.weight", W)
    if h["proj_bias"]:
        b = z(rows.numel())
        if bt is not None:
            b[used] = bt[rows[used]]
        put(pre + "in_proj.bias", b)
    # conv over [x | B | C]: the same rows minus the gate block
    crow = rows[Ih:2 * Ih + 2 * Gh * Nh] - It
    cu = crow >= 0
    C = z(crow.numel(), 1, Kh)
    C[cu, :, Kh - Kt:] = T["conv1d.weight"][crow[cu]]
    put(pre + "conv1d.weight", C)
    if h["conv_bias"]:
        b = z(crow.numel())
        if "conv1d.bias" in T:
            b[cu] = T["conv1d.bias"][crow[cu]]
        put(pre + "conv1d.bias", b)
    for name in ("dt_bias", "A_log", "D"):
        put(pre + name, T[name][head_src])
    put(pre + "norm.weight", T["norm.weight"][src])
    O = z(D, Ih)
    O[:, first] = lift(T["out_proj.weight"][:, src[first]].T, wf).T
    put(pre + "out_proj.weight", O)
    if h["proj_bias"]:
        put(pre + "out_proj.bias", lift(T["out_proj.bias"], wf) if "out_proj.bias" in T else z(D))


def _eps(c, fam):
    return float(getattr(c, fam.eps_key)) if fam.eps_key else fam.eps_fixed


def _norm_kind(c, fam):
    if fam.name == "opt" and not getattr(c, "layer_norm_elementwise_affine", True):
        return "ln_none"
    return fam.norm


def _rot(c, fam):
    """Rotated dims per head (0 for learned / no positions)."""
    if fam.rope == "none" or getattr(c, "position_embedding_type", "rope") == "nope":
        return 0
    return int(_head_dim(c) * float(getattr(c, "partial_rotary_factor", 1.0) or 1.0))


def _rope(c):
    rs = getattr(c, "rope_scaling", None) or {}
    typ = rs.get("rope_type", rs.get("type", "default")) or "default"
    return typ, rs


def _parallel(c, fam):
    if fam.block == "falcon":
        return bool(c.new_decoder_architecture or c.parallel_attn)
    return fam.block == "shared" or (fam.block in ("neox", "shared_cfg")
                                     and bool(getattr(c, "use_parallel_residual", False)))


def _one_norm(c, fam):
    """A parallel block whose attention and FFN branches read one norm."""
    if fam.block == "falcon":
        return _parallel(c, fam) and not (c.new_decoder_architecture and (c.num_ln_in_parallel_attn or 2) == 2)
    return fam.block == "shared" or (fam.block == "shared_cfg" and _parallel(c, fam))


def _windows(c, fam):
    """Per-layer attention window (None = full) as the stock class builds it."""
    n = c.num_hidden_layers
    sw = getattr(c, "sliding_window", None)
    if fam.window == "global":
        return [sw] * n
    if fam.window == "layers":
        lt = getattr(c, "layer_types", None) or ["full_attention"] * n
        return [sw if t == "sliding_attention" else None for t in lt]
    if fam.window == "neo":                                    # GPT-Neo: attention_layers global / local, window_size
        return [c.window_size if t == "local" else None for t in c.attention_layers]
    if fam.window == "flag":                                   # the eager mask: every layer (see FAMILIES)
        return [sw if getattr(c, "use_sliding_window", False) and sw is not None else None] * n
    return [None] * n


def _qkn(c, fam):
    """The family's per-head q_norm / k_norm modules are built (a flag or a config key)."""
    return _src(c, fam.qk_norm)


def _attn_bias(c, fam):
    """(q/k/v bias, o bias)."""
    return _src(c, fam.bias[0]), _src(c, fam.bias[1])


def _mlp_bias(c, fam):
    return _src(c, fam.mlp_bias)


@dataclass(frozen=True)
class MoE:
    """One MoE layer: E experts of width ew, top-k, renormalised gates (norm), a shared expert of width sw
    (gated by a sigmoid: qwen), and whether a selection bias (ERNIE's moe_statics) picks the experts."""
    ew: int
    E: int
    k: int
    norm: bool
    sw: int = 0
    shared_gate: bool = False
    sel_bias: bool = False
    score: str = "softmax"       # the router score: "softmax" | "sigmoid" (GLM-4.5: sigmoid, weights not divided by m)
    scale: float = 1.0           # routed_scaling_factor (the routed experts' weights x scale)
    groups: tuple = (1, 1)       # (n_group, topk_group): group-limited routing (active when 1 < topk_group < n_group)


def _ffn_layers(c, fam):
    """Per layer: ('dense', width) or ('moe', MoE)."""
    n = c.num_hidden_layers
    if fam.name == "granitemoehybrid" and getattr(c, "num_local_experts", 0):
        # Granite-4.0-H tiny / small: routed experts (softmax over the top-k logits) beside the shared MLP
        return [("moe", MoE(c.intermediate_size, c.num_local_experts, c.num_experts_per_tok, True,
                            c.shared_intermediate_size))] * n
    if fam.moe == "olmoe":
        return [("moe", MoE(c.intermediate_size, c.num_experts, c.num_experts_per_tok, bool(c.norm_topk_prob)))] * n
    if fam.moe == "phimoe":                         # PhiMoE: SparseMixer top-2 (eval: two masked softmaxes)
        return [("moe", MoE(c.intermediate_size, c.num_local_experts, 2, False, score="sparsemixer"))] * n
    if fam.moe in ("mixtral", "granite"):          # Granite-MoE: softmax over the top-k logits = renormalised
        return [("moe", MoE(c.intermediate_size, c.num_local_experts, c.num_experts_per_tok, True))] * n
    if fam.moe == "qwen":
        out = []
        for i in range(n):
            sparse = (i not in (getattr(c, "mlp_only_layers", None) or [])
                      and c.num_experts > 0 and (i + 1) % getattr(c, "decoder_sparse_step", 1) == 0)
            if sparse:
                sw = getattr(c, "shared_expert_intermediate_size", 0) or 0
                out.append(("moe", MoE(c.moe_intermediate_size, c.num_experts, c.num_experts_per_tok,
                                       bool(getattr(c, "norm_topk_prob", False)), sw, sw > 0)))
            else:
                out.append(("dense", c.intermediate_size))
        return out
    if fam.moe == "glm4":
        G, g = int(getattr(c, "n_group", 1) or 1), int(getattr(c, "topk_group", 1) or 1)
        spec = MoE(c.moe_intermediate_size, c.n_routed_experts, c.num_experts_per_tok, bool(c.norm_topk_prob),
                   c.moe_intermediate_size * c.n_shared_experts, False, True, "sigmoid",
                   float(c.routed_scaling_factor), (G, g))
        return [("dense", c.intermediate_size) if i < c.first_k_dense_replace else ("moe", spec) for i in range(n)]
    if fam.moe == "ernie":
        out = []
        for i in range(n):
            if ((i + 1) % c.moe_layer_interval == 0 and c.moe_layer_start_index <= i <= c.moe_layer_end_index):
                out.append(("moe", MoE(c.moe_intermediate_size, c.moe_num_experts, c.moe_k, True,
                                       c.moe_intermediate_size * c.moe_num_shared_experts, False, True)))
            else:
                out.append(("dense", c.intermediate_size))
        return out
    return [("dense", _inter(c, fam))] * n


def _act(c, fam):
    if fam.name == "lfm2":
        return "silu"                                          # Lfm2MLP: F.silu in code
    return getattr(c, fam.act_key, None) or getattr(c, "hidden_act", None)


def _softcaps(c):
    return getattr(c, "attn_logit_softcapping", None), getattr(c, "final_logit_softcapping", None)


def _stream_style(tk, hk):
    """The residual stream map for target / host norm kinds (None: not built)."""
    if tk == "rms" and hk == "rms":
        return "zero"
    if tk != "rms" and hk != "rms":
        return "mean"
    if tk == "rms":
        return "mirror"
    return None


# --- the construction plan (config level) -------------------------------------------------------------

@dataclass
class Construction:
    tfam: Fam
    hfam: Fam
    d: int
    D: int
    qmap: dict                    # target q head -> (host q head, block)
    kvmap: dict                   # host kv head -> [(target kv head, block), ...] (several: KV-head packing)
    dmaps: list                   # per block: target head dim index -> host head dim index
    rope_note: str
    units: list                   # per target layer: [(host layer, "both" | "attn" | "ffn"), ...]
    host_layer_types: list | None
    host_window: int | None
    ffn: list                     # per host layer, from _ffn_layers
    ffn_mode: dict                # host layer -> "dense" | "tied" | "shared" | "split" | "moe"
    style: str = "zero"           # residual stream map U: "zero" | "mean" | "mirror"
    a: float = 1.0                # host stream = a U x
    moe_m: dict = field(default_factory=dict)   # host layer -> expert duplication m (MoE target)
    config_changes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    table: dict | None = None     # the longrope per-pair table: {"q": blocks per host head, "packed": bool}
    extras: dict = field(default_factory=dict)   # further construction facts (see plan_construction)

    @property
    def layer_pos(self):
        """target layer i -> its first host layer (v1's layer_pos)."""
        return [u[0][0] for u in self.units]


def head_maps(Ht, KVt, Hh, KVh, q=1):
    """Target q head -> (host q head, block) and host kv head -> [(target kv head, block), ...] (HF repeat_kv:
    query head j reads KV head j // g). The target's query heads are laid end to end (grouped by KV head) and each
    host KV head takes the longest run of <= g_h of them from <= q distinct target KV heads, which sit side by side
    in it as blocks 0, 1, ... (KV-head PACKING, graph/model_graph.py kv_hosts_needed; q = 1 tiles each target KV
    head over ceil(g_t / g_h) host KV groups, its K and V replicated). Raises Refused when the host has too few."""
    gt, gh = Ht // KVt, Hh // KVh
    qmap, kvmap = {}, {}
    y, m, end = 0, 0, KVt * gt
    while y < end:
        stop = min(y + gh, (y // gt + q) * gt, end)
        if m >= KVh:
            from graph.model_graph import kv_hosts_needed
            raise Refused(OUT_OF_SCOPE, f"heads: {KVt} target KV heads x {gt} queries need "
                                        f"{kv_hosts_needed(KVt, gt, gh, q)} host KV groups of {gh}"
                                        + (f" ({q} target KV heads packed per host head)" if q > 1 else "")
                                        + f"; the host has {KVh}")
        c0 = y // gt
        kvmap[m] = [(c, c - c0) for c in range(c0, (stop - 1) // gt + 1)]
        for t in range(y, stop):
            qmap[t] = (m * gh + t - y, t // gt - c0)
        y, m = stop, m + 1
    return qmap, kvmap


def alloc_dims(q, hd_t, rot_t, lay_t, hd_h, rot_h, lay_h):
    """Dims of q target heads side by side in one host head whose rotary pairs take any frequency (a longrope
    table): block b's rotary pair i on host pair b * rot_t / 2 + i, the target's unrotated dims on the host's
    unrotated dims, then on the dims of host pairs left at frequency 0. Returns (per block: target dim -> host dim,
    per host pair: the target pair whose frequency it needs, or None for 0). Raises Refused."""
    pt, ph = _pairs(lay_t, rot_t), _pairs(lay_h, rot_h)
    if q * len(pt) > len(ph):
        raise Refused(OUT_OF_SCOPE, f"rope: {q} x {len(pt)} target rotary pairs do not fit in the host's {len(ph)}")
    req = [None] * len(ph)
    pool = list(range(rot_h, hd_h))
    for p in range(q * len(pt), len(ph)):
        pool += list(ph[p])
    if len(pool) < q * (hd_t - rot_t):
        raise Refused(OUT_OF_SCOPE, f"head_dim: {q} x {hd_t} target dims do not fit in the host's {hd_h}")
    dmaps, k = [], 0
    for b in range(q):
        dm = [None] * hd_t
        for i, (a, c) in enumerate(pt):
            p = b * len(pt) + i
            dm[a], dm[c] = ph[p]
            req[p] = i
        for e in range(hd_t - rot_t):
            dm[rot_t + e] = pool[k]
            k += 1
        dmaps.append(dm)
    return dmaps, req


def _pairs(layout, rot):
    """The two dims of each rotary pair (rotate_half or interleaved), in frequency order."""
    if layout == "inter":
        return [(2 * i, 2 * i + 1) for i in range(rot // 2)]
    return [(i, i + rot // 2) for i in range(rot // 2)]


def dim_map(hd_t, rot_t, lay_t, hd_h, rot_h, lay_h):
    """Target dim -> host dim inside a head, the host pair of each target pair, and the RoPE theta power
    (None: same theta). Target pair i -> host pair k i, k = rot_h // rot_t: host pair k i has frequency
    theta'^(-2 k i / rot_h) = theta^(-2i / rot_t) with theta' = theta^(rot_h / (k rot_t)) -- the same theta when
    rot_h = k rot_t. Unrotated dims onto the host's unrotated dims, in order. Raises Refused."""
    if rot_t == 0 and rot_h == 0:
        if hd_h < hd_t:
            raise Refused(OUT_OF_SCOPE, f"head_dim: host {hd_h} < target {hd_t}")
        return list(range(hd_t)), [], None
    if rot_t == 0 or rot_h == 0:
        raise Refused(OUT_OF_SCOPE, "positions: one side rotates (RoPE) and the other has none / learned "
                                    "absolute positions")
    if rot_h < rot_t:
        raise Refused(OUT_OF_SCOPE, f"rope: the host rotates {rot_h} dims per head, the target {rot_t} (each target "
                                    f"pair needs its own host pair)")
    if hd_h - rot_h < hd_t - rot_t:
        raise Refused(OUT_OF_SCOPE, f"rope: the target's {hd_t - rot_t} unrotated dims per head do not fit in the "
                                    f"host's {hd_h - rot_h}")
    k = rot_h // rot_t
    power = None if rot_h == k * rot_t else rot_h / (k * rot_t)
    pt, ph = _pairs(lay_t, rot_t), _pairs(lay_h, rot_h)
    dm = [None] * hd_t
    hpair = []
    for i, (a, b) in enumerate(pt):
        ha, hb = ph[k * i]
        dm[a], dm[b] = ha, hb
        hpair.append(k * i)
    for e in range(hd_t - rot_t):
        dm[rot_t + e] = rot_h + e
    return dm, hpair, power


def _inv_freq(c):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    typ, _ = _rope(c)
    with warnings_off():
        inv, scale = ROPE_INIT_FUNCTIONS[typ](c, "cpu")
    return inv.to(F64), float(scale)


def _units(t_par, h_par, L_t):
    """The host-layer units of each target layer: one layer, or (seq target on a parallel host) attention
    then FFN on two layers."""
    if t_par or not h_par:
        return [["both"] for _ in range(L_t)]
    return [["attn", "ffn"] for _ in range(L_t)]


def _place(kinds, ok, L_h):
    """Order-preserving placement of the units (flattened) on the host's layers. ok(unit index, host layer).
    Spread evenly when every unit fits everywhere, else first fit."""
    n = len(kinds)
    if n > L_h:
        return None
    if all(ok(u, j) for u in range(n) for j in range(L_h)):
        return [u * L_h // n for u in range(n)]
    pos, j = [], 0
    for u in range(n):
        while j < L_h and not ok(u, j):
            j += 1
        if j == L_h:
            return None
        pos.append(j)
        j += 1
    return pos


def _groups_on(m):
    """Group-limited routing restricts the choice (1 < n_group, topk_group < n_group)."""
    return m.groups[0] > 1 and m.groups[1] < m.groups[0]


def _moe_fit(t, h, const_ok=False):
    """Expert duplication m for an MoE target layer t in an MoE host layer h, or the reason it does not fit. The m
    copies of a target expert share its gate (softmax: equally; sigmoid: each the full score, the down projection
    / m), so they may each hold a slice of its units with the down projection x m (m e_h >= e_t). Spare host experts
    are kept out by a selection bias or, const_ok, by a router row reading a constant residual channel at -R (a
    free hidden dim). routed_scaling_factor folds into the down projections."""
    if t.score != h.score:
        return None, f"router score: target {t.score}, host {h.score}"
    if h.k % t.k:
        return None, f"top-k {h.k} is not a multiple of the target's {t.k}"
    m = h.k // t.k
    if m * h.ew < t.ew:
        return None, f"expert width {m} x {h.ew} < the target's {t.ew}"
    if m * t.E > h.E:
        return None, f"{h.E} experts cannot hold {t.E} target experts x {m} copies (top-k {t.k} -> {h.k})"
    if t.sel_bias:
        if not h.sel_bias:
            return None, "the target's selection bias needs a host with one"
        if t.score == "softmax" and not (m == 1 and h.E == t.E):
            return None, ("the target's selection bias is added to softmax weights, which copies divide: the same "
                          "experts and top-k only")
    if _groups_on(t) and not (m == 1 and h.E == t.E and h.groups[0] > 1):
        return None, "group-limited routing: the same experts and top-k in a host with groups"
    if t.norm != h.norm:
        return None, f"gates renormalised: target {t.norm}, host {h.norm}"
    if m * t.E < h.E:
        if not (h.sel_bias or const_ok):
            return None, (f"{h.E - m * t.E} spare experts and no selection bias or free hidden dim to keep them out "
                          f"(a linear router cannot mask an expert)")
        if not t.norm and t.score == "softmax":
            return None, "spare experts: the target's softmax gates are not renormalised (exactly m x its experts)"
    if t.sw:
        if h.sw < t.sw:
            return None, f"shared expert {h.sw} < the target's {t.sw}"
        if t.shared_gate and not h.shared_gate:
            return None, "the target's shared expert is gated, the host's is not"
    return m, None


def _gate0(m, selected_only=False):
    """The weight of each of the k experts a zero router (scores tied) selects: softmax 1/E (renormalised 1/k; 1/k
    too when the others' weight underflows, selected_only), sigmoid 1/2 (renormalised 1/k); x routed_scaling_factor."""
    if m.score == "sparsemixer":
        # a zero router ties every score: the first pick (expert 0) weighs 1/E, the second (expert 1, the first
        # masked to -inf) 1/(E-1); with the constant channel (2R, R) the threshold masks the rest: 1 and 1
        return 1.0 if selected_only else (1.0 / m.E + 1.0 / (m.E - 1)) / 2
    if m.norm:
        g = 1.0 / m.k
    elif m.score == "sigmoid":
        g = 0.5
    else:
        g = 1.0 / (m.k if selected_only else m.E)
    return g * m.scale


def _dense_mode(t_kind, It, f, hfam, h_mlp_bias, t_mlp_bias=False, const_ok=False):
    """How a dense target FFN (kind 'glu' / 'mlp', width It) sits in host layer f, or None."""
    if f[0] == "dense":
        if f[1] < It:
            return None
        return "dense" if t_kind == hfam.ffn else None
    if t_kind != "glu" or t_mlp_bias:
        return None
    m = f[1]
    if m.sw >= It:
        return "shared"
    if m.ew >= It:
        return "tied"
    if (m.sel_bias or const_ok) and m.k * m.ew + m.sw >= It:
        return "split"
    return None


# LIMIT constructions the script builds, for the edges the relation itself labels a limit (graph.edge_plan slot
# rows that are routes with exact = False): (slot, target, host). ReLU <- GeLU (erf or tanh): GeLU(lam u) / lam ->
# ReLU(u) as lam -> oo (|error| <= 0.17 / lam per unit), so the host's FFN input rows (and biases) are scaled by
# lam and its output columns by 1 / lam (ACT_LAMBDA).
LIMIT_BUILT = {("act", "ReLU", "GeLU"), ("act", "ReLU", "GeLU_tanh"),
               # the eps lever (graph/eps_lever.py; a payable limit from round 2, pass 5): a plain target on a host
               # whose gained RMS qk-norm is switched off -- q_proj, k_proj x s, the norm's gain sqrt(eps) / s, s chosen
               # per layer from the weights so that rho = s^2 ms(q) / eps <= 2^-60 (EPS_LEVER_RHO)
               ("qknorm", "no", "head"), ("qknorm", "no", "head_distinct"), ("qknorm", "no", "proj")}
EPS_LEVER_RHO = 2.0 ** -60
ACT_LAMBDA = 2.0 ** 20


def limit_ok(plan):
    """Every non-exact route of the edge is a limit the script builds."""
    bad = [(r["slot"], r["target"], r["host"]) for r in plan["slots"] if r["kind"] == "route" and not r["exact"]]
    return bool(bad) and all(b in LIMIT_BUILT for b in bad)


def plan_construction(traw, hraw, limit=False):
    """Config-level construction for "host emulates target", or Refused(OUT_OF_SCOPE, reasons). Collects
    every reason it can rather than stopping at the first. limit: the relation labels the edge a limit the script
    builds (LIMIT_BUILT), so those differences are built as labelled limit constructions."""
    tfam, hfam = family_of(traw, "target"), family_of(hraw, "host")
    reasons = side_support(traw, "target") or []
    reasons += side_support(hraw, "host") or []
    if tfam.layout == "mamba" or hfam.layout == "mamba":
        return _plan_mamba(traw, hraw, tfam, hfam, reasons)
    tc, hc = resolved(traw), resolved(hraw)
    changes, notes = {}, []
    # activation
    ta, ha = _act(tc, tfam), _act(hc, hfam)
    act_lam = None
    if limit and ACT_MATH.get(ta, ta) == "relu" and ACT_MATH.get(ha, ha) in ("gelu_erf", "gelu_tanh"):
        act_lam = ACT_LAMBDA
        notes.append(f"LIMIT construction (the relation labels this edge a limit): the target's ReLU on the host's "
                     f"{ha} as GeLU(lam u) / lam with lam = 2^20 (the FFN input rows and biases x lam, its output "
                     f"columns / lam; |GeLU(lam u) / lam - ReLU(u)| <= 0.17 / lam per unit): not exact")
    elif ACT_MATH.get(ta, ta) != ACT_MATH.get(ha, ha):
        reasons.append(f"activation: target {ta!r}, host {ha!r} are different functions (the script copies the FFN "
                       f"as is; the relation's act slot does not tell an erf GeLU from the tanh one)")
    # norms and the residual stream map
    tnk, hnk = _norm_kind(tc, tfam), _norm_kind(hc, hfam)
    style = _stream_style(tnk, hnk)
    if style is None:
        reasons.append("norm: a LayerNorm target on an RMSNorm host (the centring folds into the writers, but its "
                       "bias needs a bias on every reader; not built)")
    elif tnk == "ln" and hnk in ("ln_w", "ln_none"):
        reasons.append("norm: the target's LayerNorms have biases, the host's have none")
    if hnk == "ln_none" and tc.tie_word_embeddings and hc.tie_word_embeddings:
        reasons.append("norm: the host's final norm has no gain and its LM head is tied to the embedding (the "
                       "target's final gain has nowhere to go)")
    if tfam.post_only != hfam.post_only:
        reasons.append("norm placement: " + ("a post-norm-only target (x + norm(f(x))) on a pre-norm host"
                                             if tfam.post_only else "a pre-norm target on a post-norm-only host"))
    elif tfam.post_only and _resid_mult(hc, hfam) != _resid_mult(tc, tfam):
        reasons.append("norm placement: a post-norm-only pair with a residual-scale ratio (not built)")
    if _qkn(tc, tfam) and _qkn(hc, hfam) and tfam.qk_kind != hfam.qk_kind:
        reasons.append(f"qk-norm: target variant {tfam.qk_kind}, host {hfam.qk_kind}")
    if tfam.sandwich != hfam.sandwich:
        reasons.append("norm placement: " + ("the target has post-attention / post-FFN norms the host lacks"
                                             if tfam.sandwich else "the host's post-attention / post-FFN norms "
                                                                   "cannot be made the identity"))
    eps_lever = False
    if limit and _qkn(hc, hfam) and not _qkn(tc, tfam) and hfam.eps_key and not hfam.post_only:
        eps_lever = True
        notes.append("LIMIT construction (the relation labels this edge a limit): the eps lever switches the host's "
                     "q / k norm off -- q_proj, k_proj scaled by s and the norm's gain sqrt(eps) / s, s set per layer "
                     "from the weights so that rho = s^2 ms(q) / eps <= 2^-60 (the norm is then linear, to fp64 "
                     "rounding): not exact in exact arithmetic")
    elif _qkn(tc, tfam) != _qkn(hc, hfam):
        reasons.append("qk-norm: only one side has per-head q_norm / k_norm modules")
    if _softcaps(tc) != _softcaps(hc):
        reasons.append(f"softcap: target (attention, final) {_softcaps(tc)} != host {_softcaps(hc)} "
                       f"(not a setting the host may change)")
    # block
    t_par, h_par = _parallel(tc, tfam), _parallel(hc, hfam)
    if t_par and not h_par:
        reasons.append("block: a parallel-block target on a sequential host (exact only at (L+1)x width, norm-free; "
                       "not built)")
    elif t_par and _one_norm(hc, hfam) and not _one_norm(tc, tfam):
        reasons.append("block: the target's parallel branches have two norms, the host's share one")
    # biases
    tqkv, to = _attn_bias(tc, tfam)
    hqkv, ho = _attn_bias(hc, hfam)
    if tqkv and not hqkv:
        reasons.append("bias: the target's q/k/v projections have biases, the host's have none "
                       "(fine only if the target's biases are all zero)")
    if to and not ho:
        reasons.append("bias: the target's o_proj has a bias, the host's has none")
    tmb, hmb = _mlp_bias(tc, tfam), _mlp_bias(hc, hfam)
    if tmb and not hmb:
        reasons.append("bias: the target's MLP has biases, the host's has none")
    # positions
    if (tfam.rope == "none") != (hfam.rope == "none"):
        reasons.append("positions: " + ("learned absolute positions on a RoPE host" if tfam.rope == "none" else
                                        "a RoPE target on a host with learned absolute positions"))
    elif tfam.rope == "none" and hc.max_position_embeddings < tc.max_position_embeddings:
        reasons.append(f"positions: the host has {hc.max_position_embeddings} learned positions, the target "
                       f"{tc.max_position_embeddings}")
    # widths
    d, D = tc.hidden_size, hc.hidden_size
    need = 2 * d if style == "mirror" else d
    if D < need:
        reasons.append(f"hidden: host {D} < target {d}" + (" x 2 (the [h; -h] mirror: an RMSNorm target on a "
                                                           "LayerNorm host)" if style == "mirror" else ""))
    if tfam.mla or hfam.mla:
        qmap, kvmap, dmaps, table = {}, {}, [], None
        rope_note = _mla_plan(tc, tfam, hc, hfam, hraw, changes, notes, reasons)
    else:
        qmap, kvmap, dmaps, rope_note, table = _heads_rope(tc, tfam, hc, hfam, hraw, changes, notes, reasons)
    # layers and windows
    t_win, h_win = _windows(tc, tfam), _windows(hc, hfam)
    tw_set = {w for w in t_win if w is not None}
    host_lt, host_w = None, None
    units = None
    t_ffn = _target_ffn(tc, tfam)
    ffn = _ffn_layers(hc, hfam)
    modes, moe_m = {}, {}
    # a constant residual channel (a free hidden dim the embedding fills; its square folds into the host eps) lets a
    # router row select experts (+R) or keep spare ones out (-R): RMSNorm MoE hosts without a selection bias
    const_ok = (style == "zero" and D > d and bool(hfam.eps_key) and (hfam.moe in ("mixtral", "qwen", "granite", "olmoe") or hfam.name == "granitemoehybrid")
                and not hfam.sandwich and not hfam.post_only)
    # renormalisation of the top-k gates is free config where the code reads it (Qwen-MoE norm_topk_prob)
    t_norms = {f[1].norm for f in t_ffn if f[0] == "moe"}
    if hfam.moe in ("qwen", "glm4", "olmoe") and len(t_norms) == 1:
        tn = next(iter(t_norms))
        if any(f[0] == "moe" and f[1].norm != tn for f in ffn):
            changes["norm_topk_prob"] = tn
            ffn = [("moe", MoE(**{**f[1].__dict__, "norm": tn})) if f[0] == "moe" else f for f in ffn]
    # group-limited routing is config too (GLM-4.5 n_group / topk_group): the target's groups, else none
    if hfam.moe == "glm4":
        tg = {f[1].groups for f in t_ffn if f[0] == "moe" and _groups_on(f[1])}
        want = next(iter(tg)) if len(tg) == 1 else (1, 1)
        if any(f[0] == "moe" and f[1].groups != want for f in ffn):
            changes["n_group"], changes["topk_group"] = want
            ffn = [("moe", MoE(**{**f[1].__dict__, "groups": want})) if f[0] == "moe" else f for f in ffn]
    win_ok = True
    if len(tw_set) > 1:
        reasons.append(f"windows: the target uses several window sizes {sorted(tw_set)}; the host has one")
        win_ok = False
    elif tw_set and hfam.window in ("none", "flag") and any(w is None for w in t_win):
        reasons.append(f"windows: the target mixes full and sliding ({next(iter(tw_set))}) layers; host family "
                       f"{hfam.name} has no window setting (one caller-supplied mask cannot window some layers only)")
        win_ok = False
    elif tw_set and hfam.window == "global" and any(w is None for w in t_win):
        reasons.append("windows: the target mixes full and sliding layers; the host has one window for every layer")
        win_ok = False
    mask_w = None
    if win_ok:
        W = next(iter(tw_set)) if tw_set else None
        if W is not None and hfam.window in ("none", "flag"):
            # the relation's "mask" edge (window local <- full): the host attends fully and the target's window is
            # the caller's attention mask -- identical on any context <= W without one
            mask_w, W = W, None
            t_win = [None] * len(t_win)
        # the window lever: a host with a window setting takes the target's per layer (layer_types), or one window
        # for every layer -- the target's, or none (full attention: a window >= any context) for a full target
        h_win = [None] * hc.num_hidden_layers
        per = _units(t_par, h_par, tc.num_hidden_layers)
        flat = [(i, kind) for i, ks in enumerate(per) for kind in ks]
        why = {}

        hattn = _attn_layers(hc, hfam)
        t_attn = _attn_layers(tc, tfam)

        def ok(u, j):
            i, kind = flat[u]
            if kind != "ffn" and hattn[j] != t_attn[i]:
                return False                             # attention on attention, Mamba-2 on Mamba-2 (else zeroed)
            if kind != "attn":
                tk, tf = t_ffn[i]
                f = ffn[j]
                if tk == "moe":
                    if f[0] != "moe":
                        why.setdefault("moe", f"host layer {j} is dense (an MoE target layer needs an MoE host layer)")
                        return False
                    m, r = _moe_fit(tf, f[1], const_ok)
                    if m is None:
                        why.setdefault("moe", f"host layer {j}: {r}")
                        return False
                    return True
                mode = _dense_mode(tk, tf, f, hfam, hmb, tmb, const_ok)
                if mode is None:
                    why.setdefault("ffn", (j, f, tk, tf))
                    return False
            return True
        pos = _place([k for _, k in flat], ok, hc.num_hidden_layers)
        if pos is None:
            if "moe" in why:
                reasons.append(f"FFN: an MoE target layer does not fit: {why['moe']}")
            elif "ffn" in why:
                j, f, tk, It = why["ffn"]
                if f[0] == "dense":
                    reasons.append(f"FFN: host layer {j} has FFN width {f[1]} < the target's {It}" if f[1] < It else
                                   f"FFN: a {'gated (GLU)' if tk == 'glu' else 'ungated'} target FFN on a "
                                   f"{'gated' if hfam.ffn == 'glu' else 'ungated'} host FFN (not built)")
                else:
                    m = f[1]
                    reasons.append(f"FFN: host layer {j} has expert width {m.ew} and shared width {m.sw} < the "
                                   f"target's {It}" + (" (an ungated target FFN in an MoE host is not built)"
                                                       if tk == "mlp" else
                                                       " (splitting an FFN over routed experts needs a selection "
                                                       "bias that picks them for every token)"))
            else:
                n_full = sum(w is None for w in h_win)
                reasons.append(f"layers: {len(t_win)} target layers ({sum(w is None for w in t_win)} full-window)"
                               + (f" as {len(flat)} attention / FFN layers" if len(flat) > len(t_win) else "")
                               + f" do not fit in order on the host's {hc.num_hidden_layers} layers ({n_full} "
                                 f"full-window)")
        else:
            units = [[] for _ in per]
            for (i, kind), j in zip(flat, pos):
                units[i].append((j, kind))
                if kind != "attn":
                    tk, tf = t_ffn[i]
                    if tk == "moe":
                        modes[j], moe_m[j] = "moe", _moe_fit(tf, ffn[j][1], const_ok)[0]
                    else:
                        modes[j] = _dense_mode(tk, tf, ffn[j], hfam, hmb, tmb, const_ok)
            attn_layers = {j: i for i, us in enumerate(units) for j, k in us if k != "ffn"}
            if hfam.window == "global":
                if W is not None:
                    host_w = W
                    changes["sliding_window"] = W
                elif getattr(hc, "sliding_window", None) is not None:
                    changes["sliding_window"] = None
                    notes.append(f"the window lever: the host's sliding window ({hc.sliding_window}) is removed "
                                 f"(sliding_window null, full attention: a window longer than any context), so a "
                                 f"full-attention target layer runs on any host layer")
            elif hfam.window == "neo":
                own = list(hc.attention_layers)
                for j, i in attn_layers.items():
                    own[j] = "local" if t_win[i] is not None else "global"
                changes["attention_types"] = [[[t], 1] for t in own]
                changes["attention_layers"] = own             # (a saved config carries the expanded list too)
                if W is not None:
                    host_w = W
                    changes["window_size"] = W
            elif hfam.window == "flag":
                if getattr(hc, "use_sliding_window", False) and getattr(hc, "sliding_window", None) is not None:
                    changes["use_sliding_window"] = False
                    notes.append(f"the window lever: the host's own sliding window ({hc.sliding_window}, "
                                 f"use_sliding_window) is switched off, so every host layer attends fully")
            elif hfam.window == "layers":
                own = list(getattr(hc, "layer_types", None) or ["full_attention"] * hc.num_hidden_layers)
                lever = [j for j, i in attn_layers.items() if t_win[i] is None and own[j] == "sliding_attention"]
                for j, i in attn_layers.items():
                    own[j] = "sliding_attention" if t_win[i] is not None else "full_attention"
                if lever:
                    notes.append(f"the window lever: host layers {_ranges(sorted(lever))} are sliding-window layers in "
                                 f"the host's own config; they carry full-attention target layers, so layer_types "
                                 f"sets them to full attention")
                host_lt = own
                changes["layer_types"] = own
                if W is not None:
                    host_w = W
                    changes["sliding_window"] = W
                elif any(t == "sliding_attention" for t in own):
                    host_w = getattr(hc, "sliding_window", None)
                if hfam.name in ("qwen2", "qwen3"):
                    changes["use_sliding_window"] = any(t == "sliding_attention" for t in own)
                    if changes["use_sliding_window"]:
                        changes["sliding_window"] = host_w
    extras = {}
    if eps_lever:
        extras["eps_qk"] = True
    if act_lam is not None:
        extras["act_lambda"] = act_lam
    if id(changes) in _MLA_PLAN:
        extras["mla"] = _MLA_PLAN.pop(id(changes))
    if not all(_attn_layers(tc, tfam)):
        why = _conv_fit(tc, tfam, hc, hfam) if tfam.name == "lfm2" else _m2_fit(tc, tfam, hc, hfam)
        if isinstance(why, str):
            reasons.append(why)
        else:
            extras["conv" if tfam.name == "lfm2" else "m2"] = why
    if units is not None and const_ok:
        need = [j for j, md in modes.items() if md == "split" and not ffn[j][1].sel_bias]
        need += [j for j, md in modes.items() if md == "moe" and not ffn[j][1].sel_bias
                 and ffn[j][1].E > moe_m[j] * _t_experts(t_ffn, units, j)]
        if need:
            extras["const"] = sorted(need)
            notes.append(f"a constant residual channel (dim {d}, filled by the embedding; its square is taken out of "
                         f"the host eps, so every norm still sees the target's rms) feeds the routers of host layers "
                         f"{_ranges(sorted(need))}: +R selects the experts that hold a dense FFN, -R keeps spare experts "
                         f"out (R large enough that their softmax weight underflows to 0)")
    if mask_w is not None and units is not None:
        extras["mask_window"] = mask_w
        notes.append(f"the target's sliding window ({mask_w}) is the relation's mask edge: host family {hfam.name} has no "
                     f"window setting, so it attends fully and the window is the caller's attention mask (a 4-D "
                     f"mask that keeps the last {mask_w} keys); on a context of <= {mask_w} tokens no mask is needed")
    if table is not None and hfam.name in ("gemma3_text", "olmo3") and units is not None and host_lt:
        if any(host_lt[j] == "sliding_attention" for us in units for j, k in us if k != "ffn"):
            reasons.append(f"rope: {hfam.name}'s sliding layers rotate with a default-type table of their own; a "
                           f"longrope table reaches its full-attention layers only")
    if reasons:
        raise Refused(OUT_OF_SCOPE, reasons)
    # front end
    changes["vocab_size"] = tc.vocab_size
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        changes[k] = getattr(tc, k, None)
    if not tc.tie_word_embeddings and hc.tie_word_embeddings:
        changes["tie_word_embeddings"] = False
        notes.append("the target's LM head is untied and the host ties it: tie_word_embeddings set to false (the "
                     "head is part of the front end the relation ignores, like vocab_size)")
    # the residual stream: host = a U x. a = r_host / r_target (Granite's residual multipliers) lets o_proj /
    # down_proj be copied as they are; the host's norms then see a factor a sqrt(d / D) (sqrt(2d / D) for the
    # mirror) once eps' = a^2 eps (2) d / D
    n_d = (2 * d if style == "mirror" else d) / D
    # a stream norm over the host's D dims sees kappa (ms_t + eps_t) with kappa = a^2 n d / D once the host eps plus
    # the constant channel's c^2 / D is kappa eps_t
    et = _eps(tc, tfam)
    if hfam.eps_key and "m2" in extras:
        # the Mamba-2 gated RMSNorm sees m copies of the target's channels (the same rms) and shares the config eps:
        # eps' = eps, so the stream is carried x sqrt(D / d) (x sqrt(2 D / d) with the constant channel, c^2 = D eps)
        a = math.sqrt((2.0 if "const" in extras else 1.0) / n_d)
        changes[hfam.eps_key] = et
        extras["const_c2"] = D * et * (a * a * n_d - 1.0)
        notes.append(f"Mamba-2 layers: the gated norm's eps is the config eps, so it stays {et:g} and the residual "
                     f"stream is carried x {a:.6g}; each target channel is held {extras['m2']['m']} times")
    elif hfam.eps_key:
        a = _resid_mult(hc, hfam) / _resid_mult(tc, tfam)
        kappa = a * a * n_d
        changes[hfam.eps_key] = kappa * et * (CONST_EPS_FRAC if "const" in extras else 1.0)
        extras["const_c2"] = D * kappa * et * (1 - CONST_EPS_FRAC)
    else:
        # the host's eps is fixed (OLMo, OPT: 1e-5): the stream scale a absorbs the ratio instead
        a = math.sqrt(hfam.eps_fixed / (_eps(tc, tfam) * n_d))
        if a != 1:
            notes.append(f"the host's norm eps is fixed ({hfam.eps_fixed:g}): the residual stream is carried x {a:.6g} "
                         f"so that eps / a^2 is the target's eps at the host's width")
    if style != "zero" and D > d:
        notes.append({"mean": "LayerNorm host: the padded residual dims carry the target stream's mean, so the "
                              "host's LayerNorm centres exactly as the target's",
                      "mirror": "LayerNorm host for an RMSNorm target: the residual stream is carried as [h; -h], "
                                "whose mean is 0, so the LayerNorm is the RMSNorm"}[style])
    return Construction(tfam, hfam, d, D, qmap, kvmap, dmaps, rope_note, units, host_lt, host_w, ffn, modes,
                        style, a, moe_m, changes, notes, table, extras)


def _heads_rope(tc, tfam, hc, hfam, hraw, changes, notes, reasons):
    """The head maps and the per-head dim maps, and the host's RoPE config. First the plain layout (each target
    KV head tiled over host KV groups, target pair i on host pair k i, the host's own schedule at a theta that
    gives the target's frequencies: bit-identical when k is an integer); else, on a host whose rotary reads a
    longrope config (graph/model_graph.py ROPE_PAIRS_FREE_TYPES), a per-pair table: KV-head packing (several
    target KV heads side by side in one host head, each block rotating with the target's frequencies), and the
    target's unrotated dims on host pairs at frequency 0. Returns (qmap, kvmap, dmaps, rope_note, table) with
    the reasons appended on failure."""
    Ht, KVt, hdt = tc.num_attention_heads, _n_kv(tc), _head_dim(tc)
    Hh, KVh, hdh = hc.num_attention_heads, _n_kv(hc), _head_dim(hc)
    rot_t, rot_h = _rot(tc, tfam), _rot(hc, hfam)
    plain = []
    qmap = kvmap = dmap = None
    try:
        qmap, kvmap = head_maps(Ht, KVt, Hh, KVh)
    except Refused as e:
        plain.append(e.reasons[0])
    rope_note = ""
    if hdh < hdt:
        plain.append(f"head_dim: host {hdh} < target {hdt}")
    else:
        try:
            dmap, hpair, power = dim_map(hdt, rot_t, tfam.rope, hdh, rot_h, hfam.rope)
        except Refused as e:
            plain.append(e.reasons[0])
            dmap = None
        if not plain and rot_t:
            ch, nt = {}, []
            tab = None
            if power is not None and _table_ok(tc, tfam, hc, hfam, rot_t, rot_h) is None and not plain:
                # a non-integer rotated-dims ratio: rather than rescale theta (theta^power, rounded in fp32 and
                # overflowing it for large ratios), set the host's pairs per pair (longrope, theta 1: exact)
                try:
                    dms1, req = alloc_dims(1, hdt, rot_t, tfam.rope, hdh, rot_h, hfam.rope)
                    tab = _rope_table(tc, hc, hfam, hraw, rot_h, req, ch, nt)
                except Refused:
                    tab = None
            if isinstance(tab, str):
                rope_note, dmap = tab, dms1[0]
                changes.update(ch)
                notes += nt
            else:
                ch, nt = {}, []
                r = _rope_plan(tc, tfam, hc, hfam, hraw, rot_t, rot_h, hpair, power, ch, nt)
                if isinstance(r, list):
                    plain += r
                else:
                    rope_note = r
                    changes.update(ch)
                    notes += nt
    if not plain:
        return qmap, kvmap, [dmap], rope_note, ({"q": 1, "packed": False} if "longrope table" in rope_note else None)
    # the longrope table: packing and / or zero-frequency pairs
    why = _table_ok(tc, tfam, hc, hfam, rot_t, rot_h)
    if why:
        reasons += plain + [why]
        return None, None, None, "", None
    gt, gh = Ht // KVt, Hh // KVh
    qmax = max(1, min(hdh // max(hdt, 1), gh, (rot_h // rot_t) if rot_t else hdh))
    if _qkn(tc, tfam) and tfam.qk_kind == "head":
        qmax = 1          # a per-head q / k norm over the host head would mix the packed blocks
    last = None
    for q in range(1, qmax + 1):
        try:
            qmap, kvmap = head_maps(Ht, KVt, Hh, KVh, q)
            dmaps, req = alloc_dims(q, hdt, rot_t, tfam.rope, hdh, rot_h, hfam.rope)
        except Refused as e:
            last = e.reasons[0]
            continue
        if rot_h:
            r = _rope_table(tc, hc, hfam, hraw, rot_h, req, changes, notes)
            if isinstance(r, list):
                reasons += plain + r
                return None, None, None, "", None
        else:
            r = "no rotary on either side"
        packed = q > 1 and any(len(v) > 1 for v in kvmap.values())
        if packed:
            notes.append(f"KV-head packing: up to {q} target KV heads side by side in one host KV head (head_dim "
                         f"{hdh} >= {q} x {hdt}); each query head reads its own block (the others are 0 in its "
                         f"query), and o_proj reads that block's slice of the values")
        return qmap, kvmap, dmaps, r, {"q": q, "packed": packed}
    reasons += plain + ([last + (" (with KV-head packing over a longrope table)" if qmax > 1 else
                                 " (with a longrope table)")] if last else [])
    if _qkn(tc, tfam) and hdh >= 2 * hdt and any("heads:" in x for x in plain):
        reasons.append("heads: KV-head packing is not exact for a per-head qk-norm target (the host's q / k norm "
                       "would take the rms over every packed block)")
    return None, None, None, "", None


def _table_ok(tc, tfam, hc, hfam, rot_t, rot_h):
    """Why the host cannot take a per-pair longrope table, or None."""
    from graph.model_graph import ROPE_PAIRS_FREE_TYPES
    if rot_t == 0 and rot_h == 0:
        return None
    if rot_h == 0:
        return "rope: the host has no rotary dims (NoPE / learned positions)"
    if hfam.name not in ROPE_PAIRS_FREE_TYPES:
        return f"rope: host family {hfam.name} has no per-pair (longrope) rotary table"
    ttyp, _ = _rope(tc)
    if ttyp in RUNTIME_ROPE:
        return f"rope: the target's {ttyp!r} scaling recomputes inv_freq at run time"
    if ttyp == "longrope" and _rope(tc)[1].get("long_factor") != _rope(tc)[1].get("short_factor"):
        omax = getattr(tc, "original_max_position_embeddings", None) or tc.max_position_embeddings
        if omax < tc.max_position_embeddings:
            return ("rope: the target's longrope switches tables past its original context (two tables; a "
                    "re-indexed copy keeps both only on the plain layout)")
    return None


def _fp32_factor(w, A):
    """An fp32 factor f with fp32(1 / fp32(f * A)) == w (the target's fp32 inverse frequency; A = the host's fp32
    base ** (2 p / dim)), searched over the ulps around 1 / (w A); else the nearest (rounding noted)."""
    f32 = torch.float32
    w, A = torch.tensor(w, dtype=f32), torch.tensor(A, dtype=f32)
    f0 = torch.tensor(1.0 / (float(w) * float(A)), dtype=f32)
    cands = [f0]
    up = dn = f0
    for _ in range(16):
        up = torch.nextafter(up, torch.tensor(float("inf"), dtype=f32))
        dn = torch.nextafter(dn, torch.tensor(0.0, dtype=f32))
        cands += [up, dn]
    for f in cands:
        if torch.equal(1.0 / (f * A), w):
            return float(f), True
    return float(f0), False


ZERO_FACTOR = 1e300
FP32_MAX = 3.4028234663852886e38
# the constant channel c^2 = (1 - CONST_EPS_FRAC) a^2 eps_t d: the host eps keeps the rest (eps' = CONST_EPS_FRAC x
# the plain construction's), so c^2 / D + eps' is the plain eps' and every norm is unchanged
CONST_EPS_FRAC = 0.5
CONST_LOGIT = 1e30          # the router logit R x (normalised constant) at the target's smallest rms          # -> inf in fp32: inv_freq = 1 / inf = 0 exactly


def _rope_table(tc, hc, hfam, hraw, rot_h, req, changes, notes):
    """Set the host's rope_scaling to a longrope table with short_factor = long_factor chosen per pair, so that
    host pair p rotates at the target's frequency req[p] (or 0 when None), bit for bit where an fp32 factor
    reaches it. Check by recomputing both inv_freq buffers. Returns the note, or a list of reasons."""
    it, st = _inv_freq(tc) if _rot_any(tc) else (torch.zeros(0, dtype=F64), 1.0)
    shape = torch.arange(0, rot_h, 2, dtype=torch.int64).float() / rot_h
    best = None
    # the host's theta is free too: theta 1 makes base ** (2p / dim) = 1, so a factor equal to the target's own fp32
    # denominator theta_t ** (2i / d) gives its inverse frequency bit for bit (the default schedule); other
    # schedules (llama3, linear, yarn) are searched at a few thetas
    for theta in (1.0, float(getattr(tc, "rope_theta", 10000.0) or 10000.0),
                  float(getattr(hc, "rope_theta", None) or getattr(hc, hfam.theta_key))):
        A = theta ** shape                                      # as _compute_longrope_parameters does, in fp32
        fac, inexact = [], 0
        for p, i in enumerate(req):
            if i is None:
                fac.append(ZERO_FACTOR)
                continue
            f, ok = _fp32_factor(float(it[i]), float(A[p]))
            fac.append(f)
            inexact += not ok
        if best is None or inexact < best[2]:
            best = (theta, fac, inexact)
        if not inexact:
            break
    theta, fac, inexact = best
    if hfam.name == "phi3":                                    # Phi3Config takes exactly type / short / long
        rs = {"type": "longrope", "short_factor": fac, "long_factor": list(fac)}
        upd = {"rope_scaling": rs, hfam.theta_key: theta}
        if st == 1.0:
            upd["original_max_position_embeddings"] = hc.max_position_embeddings
        elif _rope(tc)[0] == "longrope" and getattr(tc, "original_max_position_embeddings", None):
            upd["max_position_embeddings"] = tc.max_position_embeddings
            upd["original_max_position_embeddings"] = tc.original_max_position_embeddings
        else:
            return [f"rope: the target's attention scaling {st:g} cannot be set on a Phi-3 longrope host"]
    elif hfam.name == "phimoe":                                 # PhimoeConfig: mscales and the original length inside
        rs = {"rope_type": "longrope", "short_factor": fac, "long_factor": list(fac), "short_mscale": st,
              "long_mscale": st, "attention_factor": st,
              "original_max_position_embeddings": hc.max_position_embeddings}
        upd = {"rope_scaling": rs, hfam.theta_key: theta}
    else:
        rs = {"rope_type": "longrope", "short_factor": fac, "long_factor": list(fac), "factor": 1.0,
              "attention_factor": st}
        upd = {"rope_scaling": rs, hfam.theta_key: theta}
    if getattr(hc, "head_dim", 0) is None:                     # longrope reads config.head_dim as is (Mistral: null)
        upd["head_dim"] = _head_dim(hc)
    try:
        h2 = resolved({**hraw, **changes, **upd})
        ih, sh = _inv_freq(h2)
    except Exception as e:                                     # noqa: BLE001
        return [f"rope: the host's stock config (transformers {_tf_version()}) does not take a longrope table "
                f"({type(e).__name__}: {str(e)[:120]})"]
    if 2 * ih.numel() != rot_h:
        return [f"rope: the host's stock class rotates {rot_h} dims per head but its config's RoPE has "
                f"{2 * ih.numel()} (config head_dim inconsistent with the class)"]
    if abs(sh - st) > 1e-12 * max(1.0, abs(st)):
        return [f"rope: attention_scaling target {st:g}, host {sh:g}"]
    want = torch.tensor([0.0 if i is None else float(it[i]) for i in req], dtype=F64)
    rel = ((ih - want).abs() / want.abs().clamp_min(1e-300)).max().item() if want.numel() else 0.0
    if (ih[want == 0] != 0).any() or rel > 1e-6:
        return [f"rope: the longrope table misses the target's frequencies (max rel {rel:.1e})"]
    changes.update(upd)
    used = sum(i is not None for i in req)
    if inexact:
        notes.append(f"RoPE: {inexact} of the {used} longrope factors cannot reach the target's fp32 inverse "
                     f"frequency bit for bit; they agree to {rel:.1e} relative")
    return (f"a longrope table (short = long factor per pair): {used} host pairs at the target's frequencies, "
            f"{len(req) - used} at 0" + ("" if not used else ", bit-identical inv_freq" if rel == 0 else
                                         f", inv_freq equal to {rel:.1e}"))


def _tf_version():
    import transformers
    return transformers.__version__


def _rot_any(c):
    fam = FAMILIES.get(c.model_type)
    return fam is not None and _rot(c, fam) > 0


def _rope_plan(tc, tfam, hc, hfam, hraw, rot_t, rot_h, hpair, power, changes, notes):
    """Set the host's RoPE config so that host pair hpair[i] has the target pair i's frequency; check it by
    recomputing both inv_freq buffers. Returns the note, or a list of reasons."""
    ttyp, trs = _rope(tc)
    th = float(getattr(tc, "rope_theta", 10000.0))
    moved = power is not None or rot_h != rot_t
    if moved and ttyp in RUNTIME_ROPE:
        return [f"rope: {ttyp!r} scaling recomputes inv_freq at run time from the rotated dims; a change "
                f"({rot_t} -> {rot_h} rotated dims) cannot keep it"]
    try:
        th_h = th if power is None else th ** power
    except OverflowError:
        th_h = math.inf
    if power is not None and (not math.isfinite(th_h) or th_h > FP32_MAX ** (1.0 - 2.0 / max(rot_h, 2))):
        # transformers computes base ** (2i / dim) in fp32: theta' must keep every pair's denominator finite
        return [f"rope: theta {th:g} ^ {power:.4g} = {th_h:.3g} overflows fp32 in the host's inv_freq (and the host's "
                f"rotary has no per-pair table)"]
    rs = getattr(tc, "rope_scaling", None)
    if rs and ttyp == "longrope" and rot_h != rot_t:
        rs = dict(rs)
        for key in ("long_factor", "short_factor"):
            f = [1.0] * (rot_h // 2)
            for i, p in enumerate(hpair):
                f[p] = rs[key][i]
            rs[key] = f
        notes.append("longrope: the per-pair factors re-indexed onto the host's pairs")
    upd = {hfam.theta_key: th_h, "rope_scaling": rs}
    if ttyp not in ("default", "linear", "llama3"):      # yarn / dynamic / longrope read these too
        upd["max_position_embeddings"] = tc.max_position_embeddings
        if getattr(tc, "original_max_position_embeddings", None) is not None:
            upd["original_max_position_embeddings"] = tc.original_max_position_embeddings
    if tfam.name == "gemma3_text" or hfam.name == "gemma3_text":
        tl = getattr(tc, "rope_local_base_freq", None)
        if tfam.name == hfam.name and tl is not None:
            upd["rope_local_base_freq"] = tl if power is None else tl ** power
    # check: the host's inv_freq, recomputed from its config with these values, at the target's pairs
    h2 = resolved({**hraw, **upd})
    try:
        it, st = _inv_freq(tc)
        ih, sh = _inv_freq(h2)
    except Exception as e:                                 # noqa: BLE001
        return [f"rope: the host's RoPE cannot be set to the target's ({type(e).__name__}: {e})"]
    for who, inv, rot in (("target", it, rot_t), ("host", ih, rot_h)):
        if 2 * inv.numel() != rot:
            return [f"rope: the {who}'s stock class rotates {rot} dims per head but its config's RoPE has "
                    f"{2 * inv.numel()} (config head_dim inconsistent with the class)"]
    got = ih[torch.tensor(hpair)]
    rel = ((got - it).abs() / it.abs().clamp_min(1e-300)).max().item() if it.numel() else 0.0
    if st != sh:
        return [f"rope: attention_scaling target {st:g}, host {sh:g}"]
    if rel > 1e-6:
        return [f"rope: {ttyp!r} inv_freq at the host's {rot_h} rotated dims differs from the target's at {rot_t} "
                f"(max rel {rel:.1e}; its schedule depends on the rotated dims)"]
    changes.update(upd)
    k = rot_h // rot_t
    if power is None:
        return (f"same rotated dims ({rot_t}), theta {th:g}" if not moved else
                f"target RoPE pair i on host pair {k} i ({rot_t} -> {rot_h} rotated dims): same theta {th:g}, "
                + ("bit-identical inv_freq" if rel == 0 else f"inv_freq equal to {rel:.1e}"))
    if rel:
        notes.append("RoPE: the rotated-dims ratio is not an integer, so theta is rescaled; inv_freq is computed in "
                     "fp32 at init, so the frequencies agree to ~1e-7 relative, not bit for bit")
    return (f"target RoPE pair i on host pair {k} i ({rot_t} -> {rot_h} rotated dims, ratio {rot_h / rot_t:.4g}): "
            f"theta {th:g} -> {th_h:.6g}, the same frequencies up to the fp32 rounding of inv_freq at init "
            f"(max rel {rel:.1e})")


def _t_experts(t_ffn, units, j):
    """The expert count of the target MoE layer placed on host layer j."""
    for i, us in enumerate(units):
        if any(jj == j and k != "attn" for jj, k in us):
            return t_ffn[i][1].E
    return 0


def _target_ffn(c, fam):
    """Per target layer: ('glu' | 'mlp', width) or ('moe', MoE)."""
    out = []
    for f in _ffn_layers(c, fam):
        out.append(("moe", f[1]) if f[0] == "moe" else (fam.ffn, f[1]))
    return out


# --- the relation gate --------------------------------------------------------------------------------

def relation(traw, hraw):
    from graph.edge_plan import edge_plan
    from graph.model_graph import classify, dims_of
    A = {"slots": classify(traw), "dims": dims_of(traw)}
    B = {"slots": classify(hraw), "dims": dims_of(hraw)}
    if A["slots"] is None or B["slots"] is None:
        raise Refused(NOT_AN_EDGE, "one of the models does not classify as a decoder LM")
    return A, B, edge_plan(A, B)


def gate(traw, hraw):
    """(plan, construction): raises Refused with the exit code for a non-edge, an approximate edge, or an
    exact edge outside the script's scope."""
    A, B, plan = relation(traw, hraw)
    if not plan["ok"]:
        from graph.frontier import blockers
        why = blockers(A, B)
        raise Refused(NOT_AN_EDGE, "the relation says the host does not emulate the target"
                      + (f" ({'; '.join(why)})" if why else ""))
    if not plan["exact"]:
        if limit_ok(plan):
            return plan, plan_construction(traw, hraw, limit=True)
        from graph.edge_plan import approx_reason
        raise Refused(APPROX_EDGE, approx_reason(plan))
    return plan, plan_construction(traw, hraw)


def _structure(raw, fam, role):
    """Per-model settings of a supported family that the script does not build."""
    c = resolved(raw)
    out = []
    if fam.name == "cohere" and getattr(c, "use_qk_norm", False):
        out.append(f"{role}: Cohere with use_qk_norm (a LayerNorm qk-norm; not built)")
    if fam.name == "helium" and _head_dim(c) * c.num_attention_heads != c.hidden_size:
        out.append(f"{role}: Helium with heads x head_dim != hidden (its o_proj is hidden x hidden)")
    if fam.name == "gpt_bigcode" and getattr(c, "add_cross_attention", False):
        out.append(f"{role}: GPT-BigCode with cross-attention")
    if fam.name == "gpt2" and getattr(c, "add_cross_attention", False):
        out.append(f"{role}: GPT-2 with cross-attention")
    if fam.mult == "mup" and getattr(c, "attn_layer_indices", None):
        out.append(f"{role}: Falcon-H1 with attention on some layers only (attn_layer_indices)")
    if fam.layout in ("neox", "gpt2", "opt", "bigcode") and _head_dim(c) * c.num_attention_heads != c.hidden_size:
        out.append(f"{role}: hidden size not a multiple of the heads")
    if fam.name in ("olmo", "olmo2", "olmo3", "olmoe", "flex_olmo") and getattr(c, "clip_qkv", None) is not None:
        out.append(f"{role}: OLMo with clip_qkv (q, k, v clamped: exact only in the clamp regime)")
    if fam.name == "stablelm" and getattr(c, "qk_layernorm", False):
        out.append(f"{role}: StableLM with qk_layernorm (a per-head LayerNorm qk-norm; not built)")
    if fam.name == "granitemoehybrid":
        if getattr(c, "position_embedding_type", "rope") not in ("rope", "nope"):
            out.append(f"{role}: Granite-4.0 with position_embedding_type {c.position_embedding_type!r}")
        if role == "target" and getattr(c, "num_local_experts", 0):
            out.append(f"{role}: Granite-4.0 with routed experts beside the shared MLP (not built as a target)")
    if fam.name == "falcon":
        if getattr(c, "alibi", False):
            out.append(f"{role}: Falcon with ALiBi (ALiBi via RoPE needs a spare inv_freq = 0 pair, a buffer the "
                       f"config cannot set; not built)")
        if getattr(c, "apply_residual_connection_post_layernorm", False):
            out.append(f"{role}: Falcon with the residual taken after the layer norm")
    if fam.name == "opt":
        if not c.do_layer_norm_before:
            out.append(f"{role}: OPT with post-norm blocks (do_layer_norm_before false)")
        if c.word_embed_proj_dim != c.hidden_size:
            out.append(f"{role}: OPT with an embedding projection (word_embed_proj_dim {c.word_embed_proj_dim})")
        if getattr(c, "_remove_final_layer_norm", False):
            out.append(f"{role}: OPT without a final layer norm")
    return out


def side_support(raw, role):
    """The reasons the script refuses this model in this role whatever the other side is (family, remote
    code, a quantized target), or None."""
    try:
        fam = family_of(raw, role)
    except Refused as e:
        return e.reasons
    if role == "target" and raw.get("quantization_config"):
        q = raw["quantization_config"]
        return [f"target: a quantized checkpoint ({q.get('quant_method', '?') if isinstance(q, dict) else q}); "
                f"point the script at the full-precision repo"]
    try:
        return _structure(raw, fam, role) or None
    except Refused as e:
        return e.reasons


def support(traw, hraw, limit=False):
    """None if the script builds this pair, else the reasons (limit: the edge is a limit in LIMIT_BUILT). Config
    level only: the zoo page."""
    try:
        plan_construction(traw, hraw, limit=limit)
    except Refused as e:
        return e.reasons
    except Exception as e:                                  # an odd config: say so instead of crashing
        return [f"config not understood: {type(e).__name__}: {e}"]
    return None


# --- reading the target ---------------------------------------------------------------------------------

def _embed_scale(model, c, fam):
    if fam.name in ("gemma", "gemma2"):
        return c.hidden_size ** 0.5                         # torch.tensor(hidden ** 0.5, dtype=hidden.dtype)
    if fam.name == "gemma3_text":
        return float(model.model.embed_tokens.embed_scale)
    if fam.mult in ("granite", "mup"):
        return float(c.embedding_multiplier)
    return 1.0


def _host_embed_scale(c, fam):
    if fam.mult == "gemma":
        return c.hidden_size ** 0.5                         # in an fp64 run (default dtype fp64 at init)
    if fam.mult in ("granite", "mup"):
        return float(c.embedding_multiplier)
    return 1.0


def _logit_div(c, fam):
    """Logits = lm_head(h) / L."""
    if fam.mult == "granite":
        return float(c.logits_scaling)
    if fam.mult == "cohere":
        return 1.0 / float(c.logit_scale)
    if fam.mult == "mup":
        return 1.0 / float(c.lm_head_multiplier)
    return 1.0


def _resid_mult(c, fam):
    return float(c.residual_multiplier) if fam.mult == "granite" else 1.0


def _scaling(attn):
    """The attention score scale the module applies (older classes divide by sqrt(head_dim)); OPT multiplies
    q (bias included) by it, the same thing."""
    s = getattr(attn, "scaling", None)
    return float(s) if s is not None else 1.0 / math.sqrt(attn.head_dim)


def _gpt2_scale(c, hd, i):
    s = 1.0 / math.sqrt(hd) if c.scale_attn_weights else 1.0
    return s / (i + 1) if c.scale_attn_by_inverse_layer_idx else s


def _attn_scale(model, c, fam, s, i):
    """Layer i's score scale: score = scale (q . k)."""
    if fam.layout == "gpt2":
        return _gpt2_scale(c, _head_dim(c), i)
    if fam.layout == "gptneo":
        return 1.0                                             # GPT-Neo does not scale its scores
    return _scaling(model.get_submodule(_key(s["layers"], str(i), s["attn"])))


def _moe_prefix(fam):
    return "block_sparse_moe" if fam.moe in ("mixtral", "granite", "phimoe") or fam.name == "granitemoehybrid" \
        else "mlp"


def _read_moe(sd, pre, fam, spec):
    """An MoE block as {'router', 'experts': [(gate, up, down)], 'shared', 'shared_gate', 'sel_bias', 'spec'}."""
    E = spec.E
    if fam.moe == "granite":
        gu, dn = sd[pre + ".input_linear.weight"], sd[pre + ".output_linear.weight"]
        ex = [(gu[e].chunk(2, 0)[0], gu[e].chunk(2, 0)[1], dn[e]) for e in range(E)]
        router = sd[pre + ".router.layer.weight"]
    else:
        nm = ("w1", "w3", "w2") if fam.moe in ("mixtral", "phimoe") else ("gate_proj", "up_proj", "down_proj")
        ex = [tuple(sd[f"{pre}.experts.{e}.{n}.weight"] for n in nm) for e in range(E)]
        if f"{pre}.experts.0.gate_proj.bias" in sd:
            raise Refused(OUT_OF_SCOPE, "target: experts with biases are not built")
        router = sd[pre + ".gate.weight"]
    out = {"router": router, "experts": ex, "shared": None, "shared_gate": None, "sel_bias": None, "spec": spec}
    for sn in ("shared_expert", "shared_experts"):
        if f"{pre}.{sn}.gate_proj.weight" in sd:
            out["shared"] = tuple(sd[f"{pre}.{sn}.{n}.weight"] for n in ("gate_proj", "up_proj", "down_proj"))
    if pre + ".shared_expert_gate.weight" in sd:
        out["shared_gate"] = sd[pre + ".shared_expert_gate.weight"]
    if fam.moe == "ernie":
        out["sel_bias"] = sd[pre + ".moe_statics.e_score_correction_bias"].reshape(-1)
    if fam.moe == "glm4":
        out["sel_bias"] = sd[pre + ".gate.e_score_correction_bias"].reshape(-1)
    return out


def _qkv_layout(fam, c):
    if fam.layout == "neox" or (fam.layout == "falcon" and not c.new_decoder_architecture and not c.multi_query) \
            or (fam.layout == "bigcode" and not c.multi_query):
        return "per_head"                                  # per head [q; k; v]
    if fam.layout == "falcon" and c.new_decoder_architecture:
        return "per_group"                                 # per KV group [q x g; k; v]
    return "cat"                                           # [q; k; v]


def _split_qkv(fam, c, W, b):
    """A fused projection -> ((Wq, bq), (Wk, bk), (Wv, bv)), heads in HF repeat_kv order."""
    H, KV, hd = c.num_attention_heads, _n_kv(c), _head_dim(c)
    lay = _qkv_layout(fam, c)
    if lay == "cat":
        sizes = [H * hd, KV * hd, KV * hd]
        bs = (None,) * 3 if b is None else b.split(sizes, 0)
        return tuple(zip(W.split(sizes, 0), bs))
    g = 1 if lay == "per_head" else H // KV
    n = H if lay == "per_head" else KV
    Wv = W.reshape(n, g + 2, hd, -1)
    bv = None if b is None else b.reshape(n, g + 2, hd)
    parts = [(Wv[:, :g].reshape(n * g * hd, -1), None if bv is None else bv[:, :g].reshape(-1))]
    for t in (g, g + 1):
        parts.append((Wv[:, t].reshape(n * hd, -1), None if bv is None else bv[:, t].reshape(-1)))
    return tuple(parts)


def _merge_qkv(fam, c, q, k, v, qb, kb, vb):
    """q [H, hd, D], k / v [KV, hd, D] (and biases) -> the fused weight and bias."""
    H, KV, hd = q.shape[0], k.shape[0], q.shape[1]
    D = q.shape[-1]
    lay = _qkv_layout(fam, c)
    if lay == "cat":
        return (torch.cat([q.reshape(-1, D), k.reshape(-1, D), v.reshape(-1, D)], 0),
                torch.cat([qb.reshape(-1), kb.reshape(-1), vb.reshape(-1)]))
    g = H // KV
    W = torch.cat([q.reshape(KV, g, hd, D), k[:, None], v[:, None]], 1).reshape(-1, D)
    B = torch.cat([qb.reshape(KV, g, hd), kb[:, None], vb[:, None]], 1).reshape(-1)
    return W, B


@torch.no_grad()
def read_target(model, c, fam):
    """The target's computation with every family-specific scale folded into effective weights. Weights are
    in nn.Linear orientation ([out, in]); norms as (effective gain, bias or None)."""
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    if fam.layout == "mamba":
        head = model.get_output_embeddings()
        tied = head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        return {"sd": sd, "tied": tied}
    s = _spec(fam, c)
    conv = fam.layout == "gpt2"
    off = fam.norm_offset
    r = _resid_mult(c, fam)
    par = _parallel(c, fam)
    ffns = _ffn_layers(c, fam)
    nk = _norm_kind(c, fam)

    def lin(name):
        w = sd[name + ".weight"]
        return (w.T if conv else w), sd.get(name + ".bias")

    def nrm(name, n=c.hidden_size):
        if nk == "ln_none":
            return torch.ones(n, dtype=F64), None
        b = sd.get(name + ".bias")
        return sd[name + ".weight"].to(F64) + off, (None if b is None else b.to(F64))

    layers = []
    kinds = _attn_layers(c, fam)
    for i in range(c.num_hidden_layers):
        p = _key(s["layers"], str(i))
        at = _key(p, s["attn"])
        mam = None
        mla = None
        if fam.mla:
            q = k = v = (None, None)
            o = (torch.zeros(1, dtype=F64), None)
            mla = {k_[len(at) + 1:]: v_.to(F64) for k_, v_ in sd.items() if k_.startswith(at + ".")}
            mla["o_proj.weight"] = mla["o_proj.weight"] * r
            if "o_proj.bias" in mla:
                mla["o_proj.bias"] = mla["o_proj.bias"] * r
        if not kinds[i]:
            q = k = v = (None, None)
            o = (torch.zeros(1, dtype=F64), None)
            mx = p + "." + s["mixer"] + "."
            mam = {k_[len(mx):]: v_.to(F64) for k_, v_ in sd.items() if k_.startswith(mx)}
            mam["out_proj.weight"] = mam["out_proj.weight"] * r
            if "out_proj.bias" in mam:
                mam["out_proj.bias"] = mam["out_proj.bias"] * r
        elif fam.mla:
            pass
        elif s.get("qkv"):
            W, b = lin(_key(at, s["qkv"]))
            q, k, v = _split_qkv(fam, c, W, b)
        else:
            q, k, v = lin(_key(at, s["q"])), lin(_key(at, s["k"])), lin(_key(at, s["v"]))
        if kinds[i] and not fam.mla:
            o = lin(_key(at, s["o"]))
        ln_in = None if fam.post_only else nrm(_key(p, s["ln_in"]))
        ln_ffn = ln_in if s["ln_ffn"] is None else nrm(_key(p, s["ln_ffn"]))
        if ffns[i][0] == "moe":
            ffn = {"kind": "moe", **_read_moe(sd, _key(p, _moe_prefix(fam)), fam, ffns[i][1])}
        else:
            mp = _key(p, s["mlp"])
            if fam.ffn == "mlp":
                ffn = {"kind": "mlp", "up": lin(_key(mp, s["up"])), "down": lin(_key(mp, s["down"])),
                       # a parametrised activation (Apertus xIELU: alpha_p, alpha_n, beta, eps per layer)
                       "act": {k_[len(mp) + 8:]: v_ for k_, v_ in sd.items() if k_.startswith(mp + ".act_fn.")}}
            else:
                if s.get("gate_up"):
                    W, b = lin(_key(mp, s["gate_up"]))
                    (g, u), (gb, ub) = W.chunk(2, 0), ((None, None) if b is None else b.chunk(2, 0))
                    gate, up = (g, gb), (u, ub)
                else:
                    gate, up = lin(_key(mp, s["gate"])), lin(_key(mp, s["up"]))
                ffn = {"kind": "glu", "gate": gate, "up": up, "down": lin(_key(mp, s["down"]))}
        if r != 1.0:
            o = (o[0] * r, None if o[1] is None else o[1] * r)
            if ffn["kind"] == "moe":
                ffn["experts"] = [(g, u, dn * r) for g, u, dn in ffn["experts"]]
                if ffn["shared"] is not None:
                    g, u, dn = ffn["shared"]
                    ffn["shared"] = (g, u, dn * r)
            else:
                dn = ffn["down"]
                ffn["down"] = (dn[0] * r, None if dn[1] is None else dn[1] * r)
        qk = _qkn(c, fam) and kinds[i]
        lay = {"q": tuple(q), "k": tuple(k), "v": tuple(v), "o": o, "ffn": ffn,
               "scale": _attn_scale(model, c, fam, s, i) if kinds[i] else None,
               "ln_in": ln_in, "ln_ffn": ln_ffn, "parallel": par, "mamba": mam, "mla": mla,
               "q_norm": sd[_key(at, s["qn"], "weight")].to(F64) + off if qk else None,
               "k_norm": sd[_key(at, s["kn"], "weight")].to(F64) + off if qk else None,
               "post_attn": None, "post_ffn": None}
        if fam.sandwich or fam.post_only:
            lay.update(post_attn=nrm(_key(p, s["post_attn"]))[0], post_ffn=nrm(_key(p, s["post_ffn"]))[0])
        layers.append(lay)
    c_t, L_t = _embed_scale(model, c, fam), _logit_div(c, fam)
    S = sd[s["embed"] + ".weight"]
    head = model.get_output_embeddings()
    if getattr(head, "bias", None) is not None:
        raise Refused(OUT_OF_SCOPE, "target: an LM head with a bias")
    tied = bool(c.tie_word_embeddings) or head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    return {"layers": layers, "E": S, "c": c_t, "L": L_t, "tied": tied, "final": nrm(s["final"]),
            "pos": sd[s["pos"] + ".weight"] if s["pos"] else None, "lm": None if tied else head.weight.detach()}


# --- writing the host ---------------------------------------------------------------------------------

def host_config_dict(hraw, con):
    d = dict(hraw)
    for k in ("auto_map", "quantization_config"):
        if k in d:
            d.pop(k)
            con.notes.append(f"host config: dropped {k} (stock class, unquantized weights)")
    for k, v in con.config_changes.items():
        d[k] = v
    arch = (d.get("architectures") or [None])[0]
    if arch in STOCK_ALIASES:
        d["architectures"] = [STOCK_ALIASES[arch]]
        con.notes.append(f"host config: architectures {arch} (remote code) -> {STOCK_ALIASES[arch]}, the native "
                         f"transformers port of the same model")
    d["torch_dtype"] = "float32"
    d.pop("dtype", None)
    return d


def _meta_host(hdict):
    from transformers import AutoModelForCausalLM
    from transformers import activations
    hc = resolved(hdict)
    cls = getattr(activations, "XIELUActivation", None)        # reads its buffers with .item() at init: on CPU
    orig = cls.__init__ if cls is not None else None

    def cpu_init(self, *a, **k):
        with torch.device("cpu"):
            orig(self, *a, **k)
    try:
        if cls is not None:
            cls.__init__ = cpu_init
        with warnings_off(), torch.device("meta"):
            m = AutoModelForCausalLM.from_config(hc, attn_implementation="eager")
    finally:
        if cls is not None:
            cls.__init__ = orig
    return hc, m


def _narrow(t):
    """The narrowest float dtype that holds t exactly (bf16 / fp16 / fp32), else fp64."""
    if not t.is_floating_point():
        return t
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        n = t.to(dt)
        if torch.equal(n.to(t.dtype), t):
            return n
    return t


def _fold_gains(lay, gam, ln_kind):
    """A host whose norms have no gain (OLMo): the target's gains (x gam) go into the readers' columns."""
    lay = dict(lay)
    gi, gf = lay["ln_in"][0] * gam, lay["ln_ffn"][0] * gam
    for t in ("ln_in", "ln_ffn"):
        if lay[t][1] is not None and lay[t][1].abs().max().item() != 0:
            raise Refused(OUT_OF_SCOPE, "norm: the target's LayerNorm has a bias, the host's norms have none")
    col = lambda wb, g: (wb[0].to(F64) * g, wb[1])
    lay.update(q=col(lay["q"], gi), k=col(lay["k"], gi), v=col(lay["v"], gi))
    f = dict(lay["ffn"])
    if f["kind"] == "moe":
        f["router"] = f["router"].to(F64) * gf
        f["experts"] = [(g.to(F64) * gf, u.to(F64) * gf, dn) for g, u, dn in f["experts"]]
        if f["shared"] is not None:
            g, u, dn = f["shared"]
            f["shared"] = (g.to(F64) * gf, u.to(F64) * gf, dn)
        if f["shared_gate"] is not None:
            f["shared_gate"] = f["shared_gate"].to(F64) * gf
    else:
        f["up"] = col(f["up"], gf)
        if f["kind"] == "glu":
            f["gate"] = col(f["gate"], gf)
    lay["ffn"] = f
    one = torch.ones_like(gi)
    lay["ln_in"], lay["ln_ffn"] = (one / gam, None), (one / gam, None)
    return lay


@torch.no_grad()
def build_host(T, tc, hdict, con, log=print):
    """The host state dict (each tensor in its narrowest exact dtype) for target T = read_target(...)."""
    if con.extras.get("kind") == "mamba":
        return _build_mamba(T, tc, hdict, con)
    hc, meta = _meta_host(hdict)
    tfam, hfam = con.tfam, con.hfam
    shapes = {k: tuple(v.shape) for k, v in meta.state_dict().items()}
    tied_h = bool(hc.tie_word_embeddings)
    s = _spec(hfam, hc)
    conv = hfam.layout == "gpt2"
    hnk = _norm_kind(hc, hfam)
    if tied_h:
        shapes.pop(s["lm"] + ".weight", None)
    d, D = con.d, con.D
    Ht, KVt, hdt = tc.num_attention_heads, _n_kv(tc), _head_dim(tc)
    Hh, KVh, hdh = hc.num_attention_heads, _n_kv(hc), _head_dim(hc)
    # The host's residual stream is a U x (U: zero-padding, the mean-carrying padding or the mirror). With
    # eps' = a^2 eps n d / D (n = 2 for the mirror) every norm over it sees normalised = (1 / gam) [target
    # normalised; 0], gam = sqrt(n d / D). So a norm read by a projection takes gain x gam (its output on the
    # first d dims is the target's, unscaled), a norm whose output joins the stream (Gemma's post-norms) x a gam,
    # and every writer (embedding, positions, o_proj, down_proj) is post-composed with a U.
    a, style = con.a, con.style
    gam = math.sqrt((2 if style == "mirror" else 1) * d / D)
    const_layers = set(con.extras.get("const", ()))
    # the qk-norm shares the host eps: q, k carry sqrt(eps' / (kappa eps_t)) more (1 without the constant channel)
    cf = math.sqrt(hdict[hfam.eps_key] / (a * a * gam * gam * _eps(tc, tfam))) if hfam.eps_key else 1.0
    off = hfam.norm_offset
    r_h, c_h, L_h = _resid_mult(hc, hfam), _host_embed_scale(hc, hfam), _logit_div(hc, hfam)
    tqk = _qkn(tc, tfam)
    b = math.sqrt(hdh / hdt) if tqk else 1.0
    r_q = Ht * hdt / (Hh * hdh)                                  # target entries / host entries of q, k
    r_k = sum(len(v) for v in con.kvmap.values()) * hdt / (KVh * hdh)
    assert not hfam.post_only or a == 1.0, "post-norm-only hosts read the stream unnormalised (a = 1)"
    dms = [torch.tensor(x) for x in con.dmaps]
    dm = dms[0] if dms else None
    hqkv, ho = _attn_bias(hc, hfam)
    hmb = _mlp_bias(hc, hfam)
    # Falcon-H1's muP multipliers: attention input / key / output, MLP gate / down
    mu = {"in": 1.0, "key": 1.0, "attn_out": 1.0, "gate": 1.0, "down": 1.0}
    if hfam.mult == "mup":
        mu.update({"in": float(hc.attention_in_multiplier), "key": float(hc.key_multiplier),
                   "attn_out": float(hc.attention_out_multiplier), "gate": float(hc.mlp_multipliers[0]),
                   "down": float(hc.mlp_multipliers[1])})
    sd = {}
    hattn = _attn_layers(hc, hfam)
    no_attn = [None]
    skip_attn = [None]

    def put(key, t):
        if no_attn[0] and key.startswith(no_attn[0]) and key not in shapes:
            return                                             # a hybrid host's Mamba-2 layer has no attention
        if skip_attn[0] and key.startswith(skip_attn[0]):
            return                                             # MLA: written by _write_mla
        put_raw(key, t)

    def put_raw(key, t):
        assert key in shapes, key
        if conv and key.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            t = t.T                                            # GPT-2's Conv1D stores [in, out]
        assert tuple(t.shape) == shapes[key], (key, tuple(t.shape), shapes[key])
        sd[key] = _narrow(t.contiguous())

    def z(*shape):
        return torch.zeros(*shape, dtype=F64)

    def lift(X, f):
        """Post-compose a writer with f U: X [..., d] (residual last) -> [..., D]."""
        X = X.to(F64)
        out = z(*X.shape[:-1], D)
        out[..., :d] = X * f
        if style == "mean" and D > d:
            out[..., d:] = (X.mean(-1, keepdim=True) * f).expand(*X.shape[:-1], D - d)
        elif style == "mirror":
            out[..., d:2 * d] = -X * f
        return out

    def norm_vec(w, factor):
        v = z(D)
        v[: w.numel()] = w * factor
        return v - off                                         # stored = effective - offset

    def norm_bias(bias, factor=1.0):
        v = z(D)
        if bias is not None:
            v[:d] = bias * factor
        return v

    def rows_q(W, bias, scale):                                 # target q rows -> host [Hh, hdh, D]
        Wh, bh = z(Hh, hdh, D), z(Hh, hdh)
        Wt = W.to(F64).view(Ht, hdt, -1)
        for h, (j, blk) in con.qmap.items():
            Wh[j, dms[blk], :d] = Wt[h] * scale
            if bias is not None:
                bh[j, dms[blk]] = bias.to(F64).view(Ht, hdt)[h] * scale
        return Wh, bh

    def rows_kv(W, bias, scale):
        Wh, bh = z(KVh, hdh, D), z(KVh, hdh)
        Wt = W.to(F64).view(KVt, hdt, -1)
        for m, cs in con.kvmap.items():
            for c, blk in cs:
                Wh[m, dms[blk], :d] = Wt[c] * scale
                if bias is not None:
                    bh[m, dms[blk]] = bias.to(F64).view(KVt, hdt)[c] * scale
        return Wh, bh

    def cols_o(W, bias, f):
        Wh = z(D, Hh, hdh)
        Wt = W.to(F64).view(-1, Ht, hdt)
        L = lift(Wt.permute(1, 2, 0), f)                       # [Ht, hdt, D]
        for h, (j, blk) in con.qmap.items():
            Wh[:, j, dms[blk]] = L[h].T
        return Wh.reshape(D, Hh * hdh), (z(D) if bias is None else lift(bias, f))

    def ffn_in(W, bias, width, scale=1.0):
        Wh, bh = z(width, D), z(width)
        Wh[: W.shape[0], :d] = W.to(F64) * scale
        if bias is not None:
            bh[: W.shape[0]] = bias.to(F64) * scale
        return Wh, bh

    def ffn_out(W, bias, width, f):
        Wh = z(D, width)
        Wh[:, : W.shape[1]] = lift(W.to(F64).T, f).T
        return Wh, (z(D) if bias is None else lift(bias, f))

    con.extras["act0"] = T["layers"][0]["ffn"] if T["layers"] and T["layers"][0]["ffn"].get("act") else {}
    unit_of = {}
    for i, us in enumerate(con.units):
        for j, kind in us:
            unit_of[j] = (i, kind)
    wf = a / r_h                                               # writer factor
    for j in range(hc.num_hidden_layers):
        p = _key(s["layers"], str(j))
        at = _key(p, s["attn"])
        i, kind = unit_of.get(j, (None, None))
        lay = T["layers"][i] if i is not None else None
        if lay is not None and hnk == "ln_none":
            lay = _fold_gains(lay, gam, hnk)
        mix = lay is not None and kind != "ffn"
        att = mix and hattn[j] and not hfam.mla
        fon = lay is not None and kind != "attn"
        no_attn[0] = None if hattn[j] else at + "."
        skip_attn[0] = at + "." if hfam.mla else None
        if hfam.mla:
            _write_mla(put_raw, at + ".", lay["mla"] if mix else None, con.extras["mla"], d, D, wf, lift,
                       lay["scale"] / _scaling(meta.get_submodule(at)) if mix else 1.0, shapes)
        # attention
        if att:
            s_h = (_gpt2_scale(hc, hdh, j) if conv else 1.0 if hfam.layout == "gptneo" else
                   _scaling(meta.get_submodule(at)))
            alpha = lay["scale"] / s_h
            # qk-norm: q, k scaled by a gam b so that their rms over the padded head is a gam x the target's --
            # the same factor as the hidden norms, so one eps serves both; the gain then takes alpha / b
            if con.extras.get("eps_qk"):
                # the eps lever: |q_r| <= ||w_r . g_in|| sqrt(d) (the pre-norm's output has rms <= 1)
                eh = float(hdict[hfam.eps_key])
                g_in, b_in = lay["ln_in"][0].to(F64), lay["ln_in"][1]

                def qk_bound(W, bias):                         # |W (g x_hat + beta) + bias|, ||x_hat||^2 <= d
                    W = W.to(F64)
                    r = (W * g_in).norm(dim=1) * math.sqrt(d)
                    if b_in is not None:
                        r = r + (W @ b_in.to(F64)).abs()
                    if bias is not None:
                        r = r + bias.to(F64).abs()
                    return r.max().item()
                bnd = max(qk_bound(*lay["q"]) * alpha, qk_bound(*lay["k"]) / mu["key"], 1e-30)
                lev_s = math.sqrt(EPS_LEVER_RHO * eh) / bnd
                lev_g = math.sqrt(eh) / lev_s
                qs, ks = alpha * lev_s, lev_s / mu["key"]     # (the scales ride on the rows, the gain is shared)
            elif tqk and tfam.qk_kind == "proj":
                # one RMSNorm over the whole projection: the host's holds r = (target entries) / (host entries) of
                # it, so q, k are scaled by a gam cf / sqrt(r) and the gains by sqrt(r) (one eps serves every norm)
                qs, ks = a * gam * cf / math.sqrt(r_q), a * gam * cf / math.sqrt(r_k)
            else:
                qs, ks = (a * gam * b * cf, a * gam * b * cf) if tqk else (alpha, 1.0 / mu["key"])
            q, qb = rows_q(*lay["q"], qs / mu["in"])
            k, kb = rows_kv(*lay["k"], ks / mu["in"])
            v, vb = rows_kv(*lay["v"], 1.0 / mu["in"])
            o, ob = cols_o(*lay["o"], wf / mu["attn_out"])
            if not hqkv:
                for name, bias in (("q", lay["q"][1]), ("k", lay["k"][1]), ("v", lay["v"][1])):
                    if bias is not None and bias.abs().max().item() != 0:
                        raise Refused(OUT_OF_SCOPE, f"bias: target {name}_proj bias is nonzero; the host has none")
            if not ho and lay["o"][1] is not None and lay["o"][1].abs().max().item() != 0:
                raise Refused(OUT_OF_SCOPE, "bias: target o_proj bias is nonzero; the host has none")
        else:
            q, k, v = z(Hh, hdh, D), z(KVh, hdh, D), z(KVh, hdh, D)
            qb, kb, vb = z(Hh, hdh), z(KVh, hdh), z(KVh, hdh)
            o, ob = z(D, Hh * hdh), z(D)
        if s.get("qkv"):
            W, B = _merge_qkv(hfam, hc, q, k, v, qb, kb, vb)
            put(_key(at, s["qkv"], "weight"), W)
            if hqkv:
                put(_key(at, s["qkv"], "bias"), B)
        else:
            for n_, W, B in (("q", q, qb), ("k", k, kb), ("v", v, vb)):
                put(_key(at, s[n_], "weight"), W.reshape(-1, D))
                if hqkv:
                    put(_key(at, s[n_], "bias"), B.reshape(-1))
        put(_key(at, s["o"], "weight"), o)
        if ho:
            put(_key(at, s["o"], "bias"), ob)
        if _qkn(hc, hfam) and con.extras.get("eps_qk"):
            n_q, n_k = (Hh * hdh, KVh * hdh) if hfam.qk_kind == "proj" else (hdh, hdh)
            qn, kn = (z(n_q) + lev_g, z(n_k) + lev_g) if att else (z(n_q), z(n_k))
            put(_key(at, s["qn"], "weight"), qn - off)
            put(_key(at, s["kn"], "weight"), kn - off)
        elif _qkn(hc, hfam):
            qn, kn = z(hdh), z(hdh)
            if hfam.qk_kind == "proj":
                qn, kn = z(Hh * hdh), z(KVh * hdh)
                if att:
                    gq = lay["q_norm"].view(Ht, hdt) * (alpha * math.sqrt(r_q))
                    gk = lay["k_norm"].view(KVt, hdt) * math.sqrt(r_k)
                    for h, (jj, blk) in con.qmap.items():
                        qn[jj * hdh + dms[blk]] = gq[h]
                    for mm_, cs in con.kvmap.items():
                        for c_, blk in cs:
                            kn[mm_ * hdh + dms[blk]] = gk[c_]
            elif att:
                qn[dm] = lay["q_norm"] * (alpha / b)
                kn[dm] = lay["k_norm"] / b
            put(_key(at, s["qn"], "weight"), qn - off)
            put(_key(at, s["kn"], "weight"), kn - off)
        # norms: a norm whose branch is unused gets gain 0
        if hnk != "ln_none" and hfam.post_only:
            put(_key(p, s["post_attn"], "weight"), norm_vec(lay["post_attn"], a * gam) if att else z(D) - off)
            put(_key(p, s["post_ffn"], "weight"), norm_vec(lay["post_ffn"], a * gam) if fon else z(D) - off)
        elif hnk != "ln_none":
            def ln(name, nb, on, factor=gam):
                put(_key(p, name, "weight"), norm_vec(nb[0], factor) if on else z(D) - off)
                if hnk == "ln":
                    put(_key(p, name, "bias"), norm_bias(nb[1]) if on else z(D))
                elif on and nb[1] is not None and nb[1].abs().max().item() != 0:
                    raise Refused(OUT_OF_SCOPE, "norm: the target's LayerNorm has a bias, the host's has none")
            if s["ln_ffn"] is None:
                # one norm for both branches: in an attention-only layer it is the target's input norm, in an
                # FFN-only layer the target's FFN norm (for a one-norm target the same module)
                ln(s["ln_in"], lay and (lay["ln_in"] if att else lay["ln_ffn"]), att or fon)
            else:
                ln(s["ln_in"], lay and lay["ln_in"], mix)
                ln(s["ln_ffn"], lay and lay["ln_ffn"], fon)
                if j in const_layers:                              # the router reads the constant channel, gain 1
                    w_ = sd[_key(p, s["ln_ffn"], "weight")].to(F64).clone()
                    w_[d] = 1.0 - off
                    sd[_key(p, s["ln_ffn"], "weight")] = _narrow(w_)
            if hfam.sandwich:
                put(_key(p, s["post_attn"], "weight"), norm_vec(lay["post_attn"], a * gam) if att else z(D) - off)
                put(_key(p, s["post_ffn"], "weight"), norm_vec(lay["post_ffn"], a * gam) if fon else z(D) - off)
        # FFN
        _write_ffn(put, sd, shapes, p, s, hfam, con, j, lay if fon else None, ffn_in, ffn_out, z, wf, mu, D, hmb)
        # Falcon-H1: the Mamba-2 branch beside attention is zeroed (in_proj and out_proj 0: its output is 0)
        # the Mamba-2 branch / layer (Falcon-H1 beside attention, Bamba / Granite-4.0-H in their own layers) zeroed:
        # in_proj and out_proj 0, so its output is 0
        if mix and not hattn[j]:
            if "m2" in con.extras:
                _write_m2(put, p + "." + s["mixer"] + ".", lay["mamba"], con.extras["m2"], d, D, wf, a, lift)
            else:
                _write_shortconv(put, p + "." + s["mixer"] + ".", lay["mamba"], con.extras["conv"], d, D, wf, lift)
        else:
            for key in shapes:
                if key.startswith(p + "." + s["mixer"] + "."):
                    sd[key] = torch.zeros(shapes[key], dtype=torch.bfloat16)
    # front end
    V = tc.vocab_size
    emb = lift(T["E"].to(F64) * T["c"], a / c_h)
    if const_layers:                                           # c^2 = (1 - frac) a^2 eps_t d, in every token's row
        emb[:, d] = math.sqrt(con.extras["const_c2"]) / c_h
    put(s["embed"] + ".weight", emb)
    if s["pos"]:
        # OPT's learned table is indexed from 2 (OPTLearnedPositionalEmbedding offset), GPT-2's / GPT-Neo's from 0:
        # position p reads row p + offset on each side
        P = z(*shapes[s["pos"] + ".weight"])
        ot, oh = POS_OFFSET.get(tfam.name, 0), POS_OFFSET.get(hfam.name, 0)
        src = T["pos"][ot:]
        n_ = min(src.shape[0], P.shape[0] - oh)
        P[oh:oh + n_] = lift(src[:n_], a)
        put(s["pos"] + ".weight", P)
    gamma, beta = T["final"]
    if hnk == "ln_none":
        if tied_h:
            raise Refused(OUT_OF_SCOPE, "the host's final norm has no gain and its head is tied to the embedding")
        if beta is not None and beta.abs().max().item() != 0:
            raise Refused(OUT_OF_SCOPE, "norm: the target's final LayerNorm has a bias, the host's norms have none")
        Eout = (T["E"].to(F64) if T["tied"] else T["lm"].to(F64)) / T["L"]
        lm = z(V, D)
        lm[:, :d] = Eout * L_h * gamma * gam
        put(s["lm"] + ".weight", lm)
    elif tied_h:
        if not T["tied"]:
            raise Refused(OUT_OF_SCOPE, "the host ties its LM head to the embedding and the target does not")
        kappa = c_h * L_h / (a * T["c"] * T["L"])                # E_out = E_in / (c_t L_t); host head = emb
        put(s["final"] + ".weight", norm_vec(gamma, kappa * gam))
        if hnk == "ln":
            put(s["final"] + ".bias", norm_bias(beta, kappa))
        elif beta is not None and beta.abs().max().item() != 0:
            raise Refused(OUT_OF_SCOPE, "norm: the target's final LayerNorm has a bias, the host's has none")
    else:
        Eout = (T["E"].to(F64) if T["tied"] else T["lm"].to(F64)) / T["L"]
        lm = z(V, D)
        lm[:, :d] = Eout * L_h
        put(s["lm"] + ".weight", lm)
        put(s["final"] + ".weight", norm_vec(gamma, gam))
        if hnk == "ln":
            put(s["final"] + ".bias", norm_bias(beta))
        elif beta is not None and beta.abs().max().item() != 0:
            raise Refused(OUT_OF_SCOPE, "norm: the target's final LayerNorm has a bias, the host's has none")
    if s["lm"] + ".bias" in shapes and s["lm"] + ".bias" not in sd:     # PhiMoE lm_head_bias: the target has none
        put(s["lm"] + ".bias", z(shapes[s["lm"] + ".bias"][0]))
    missing = set(shapes) - set(sd)
    if missing:
        raise RuntimeError(f"host keys not written: {sorted(missing)[:8]}")
    return sd


def _write_ffn(put, sd, shapes, p, s, hfam, con, j, lay, ffn_in, ffn_out, z, wf, mu, D, hmb):
    f = con.ffn[j]
    mode = con.ffn_mode.get(j) if lay is not None else None
    tf = lay["ffn"] if lay is not None else None
    d = con.d
    if f[0] == "dense":
        width = f[1]
        if mode == "dense":
            lam = con.extras.get("act_lambda") or 1.0          # the ReLU <- GeLU limit (LIMIT_BUILT)
            if tf["kind"] == "glu":
                g, gb = ffn_in(*tf["gate"], width, lam / mu["gate"])
            u, ub = ffn_in(*tf["up"], width, 1.0 if tf["kind"] == "glu" else lam)
            dn, dnb = ffn_out(tf["down"][0] / lam, tf["down"][1], width, wf / mu["down"])   # (its bias unscaled)
        else:
            g, gb = z(width, D), z(width)
            u, ub, dn, dnb = z(width, D), z(width), z(D, width), z(D)
        mp = _key(p, s["mlp"])
        if hfam.ffn == "mlp":
            pairs = [(s["up"], u, ub), (s["down"], dn, dnb)]
        elif s.get("gate_up"):
            pairs = [(s["gate_up"], torch.cat([g, u], 0), torch.cat([gb, ub])), (s["down"], dn, dnb)]
        else:
            pairs = [(s["gate"], g, gb), (s["up"], u, ub), (s["down"], dn, dnb)]
        for n_, W, B in pairs:
            put(_key(mp, n_, "weight"), W)
            if hmb:
                put(_key(mp, n_, "bias"), B)
        for key in [k for k in shapes if k.startswith(mp + ".act_fn.")]:
            # the activation's own parameters: the target layer's (a zero-padded unit's output meets a zero
            # down_proj column, whatever act(0) is); an identity layer keeps the first target layer's
            src = (tf if (mode == "dense" and tf) else con.extras.get("act0") or {}).get("act", {})
            name = key[len(mp) + 8:]
            if name not in src:
                raise Refused(OUT_OF_SCOPE, f"FFN: the host's activation has parameters ({name}) the target lacks")
            put(key, src[name].to(F64).reshape(shapes[key]))
        return
    m = f[1]
    base = _key(p, _moe_prefix(hfam))
    ew, E = m.ew, m.E
    R = CONST_LOGIT / math.sqrt(max(con.extras.get("const_c2", 1.0), 1e-300))   # router weight on the constant channel
    nar = lambda ts: tuple(_narrow(t) for t in ts)               # one narrowed copy, shared where repeated
    experts = [nar((z(ew, D), z(ew, D), z(D, ew)))] * E
    router = z(E, D)
    sel = z(E)
    shared = None
    shared_gate = z(1, D)
    if mode == "tied":
        mult = 1.0 / (m.k * _gate0(m))                           # k gates of a zero router, summed
        experts = [nar((ffn_in(tf["gate"][0], None, ew)[0], ffn_in(tf["up"][0], None, ew)[0],
                        ffn_out(tf["down"][0], None, ew, wf * mult)[0]))] * E
    elif mode == "shared":
        mult = 2.0 if m.shared_gate else 1.0                      # sigmoid(0) = 1/2
        shared = (ffn_in(tf["gate"][0], None, m.sw)[0], ffn_in(tf["up"][0], None, m.sw)[0],
                  ffn_out(tf["down"][0], None, m.sw, wf * mult)[0])
    elif mode == "split":                                         # dealt over the shared expert and the k experts
        # the selection bias (ERNIE) or the constant channel (+R) picks experts 0..k-1; the router is otherwise 0,
        # so the k gates are equal: 1/k each (renormalised, or the others' softmax weight underflows to 0)
        if m.sel_bias:
            sel[: m.k] = 1.0
        elif m.score == "sparsemixer":
            router[0, d], router[1, d] = 2 * R, R
        else:
            router[: m.k, d] = R
        Wg, Wu, Wd = tf["gate"][0], tf["up"][0], tf["down"][0]
        lo = 0
        if m.sw:
            hi = min(Wg.shape[0], m.sw)
            mult = 2.0 if m.shared_gate else 1.0
            shared = (ffn_in(Wg[:hi], None, m.sw)[0], ffn_in(Wu[:hi], None, m.sw)[0],
                      ffn_out(Wd[:, :hi], None, m.sw, wf * mult)[0])
            lo = hi
        e = 0
        while lo < Wg.shape[0]:
            hi = min(Wg.shape[0], lo + ew)
            experts[e] = nar((ffn_in(Wg[lo:hi], None, ew)[0], ffn_in(Wu[lo:hi], None, ew)[0],
                              ffn_out(Wd[:, lo:hi], None, ew, wf / _gate0(m, True))[0]))
            lo, e = hi, e + 1
    elif mode == "moe":
        mm = con.moe_m[j]
        te = tf["experts"]
        Et = len(te)
        ts = tf["spec"]
        # each copy's weight: the target's / m (softmax, or renormalised), the target's own score (unrenormalised
        # sigmoid: the copies' sum is m x it); the routed scaling factors' ratio on top
        cw = (1.0 / mm if (m.score == "sigmoid" and not m.norm) else 1.0) * ts.scale / m.scale
        for e in range(Et):
            g_, u_, dn_ = te[e]
            if g_.shape[0] <= ew:                                 # each copy holds the whole expert
                ex = nar((ffn_in(g_, None, ew)[0], ffn_in(u_, None, ew)[0], ffn_out(dn_, None, ew, wf * cw)[0]))
                exs = [ex] * mm
            else:                                                 # copy c holds slice c, its down projection x m
                step = -(-g_.shape[0] // mm)
                exs = [nar((ffn_in(g_[lo:lo + step], None, ew)[0], ffn_in(u_[lo:lo + step], None, ew)[0],
                            ffn_out(dn_[:, lo:lo + step], None, ew, wf * cw * mm)[0]))
                       for lo in range(0, g_.shape[0], step)]
                exs += [nar((z(ew, D), z(ew, D), z(D, ew)))] * (mm - len(exs))
            for c in range(mm):
                experts[e * mm + c] = exs[c]
                router[e * mm + c, :d] = tf["router"].to(F64)[e]
        if tf["sel_bias"] is not None:                            # the target's bias on each copy
            sel[:mm * Et] = tf["sel_bias"].to(F64).repeat_interleave(mm)
        if mm * Et < E:                                           # spare experts: never selected
            if m.sel_bias:
                sel[mm * Et:] = (sel[:mm * Et].min().item() if tf["sel_bias"] is not None else 0.0) - 2.0
            else:
                router[mm * Et:, d] = -R
        if tf["shared"] is not None:
            sg, su, sdn = tf["shared"]
            mult = 2.0 if (m.shared_gate and tf["shared_gate"] is None) else 1.0
            shared = (ffn_in(sg, None, m.sw)[0], ffn_in(su, None, m.sw)[0], ffn_out(sdn, None, m.sw, wf * mult)[0])
            if tf["shared_gate"] is not None:
                shared_gate[0, :d] = tf["shared_gate"].to(F64)[0]
    if hfam.moe == "granite" or hfam.name == "granitemoehybrid":
        put(base + ".router.layer.weight", router)
        gu = torch.stack([torch.cat([g.to(F64), u.to(F64)], 0) for g, u, _ in experts])
        put(base + ".input_linear.weight", gu)
        put(base + ".output_linear.weight", torch.stack([dn.to(F64) for _, _, dn in experts]))
        if hfam.name == "granitemoehybrid":                       # the shared MLP: fused [gate; up], output
            if shared is None:
                shared = (z(m.sw, D), z(m.sw, D), z(D, m.sw))
            put(_key(p, s["mlp"], s["gate_up"], "weight"), torch.cat([shared[0].to(F64), shared[1].to(F64)], 0))
            put(_key(p, s["mlp"], s["down"], "weight"), shared[2].to(F64))
        return
    put(base + ".gate.weight", router)
    nm = ("w1", "w3", "w2") if hfam.moe in ("mixtral", "phimoe") else ("gate_proj", "up_proj", "down_proj")
    for e in range(E):
        for n_, t in zip(nm, experts[e]):
            key = f"{base}.experts.{e}.{n_}.weight"
            assert key in shapes and tuple(t.shape) == shapes[key], key
            sd[key] = t
            if key[:-6] + "bias" in shapes:
                sd[key[:-6] + "bias"] = torch.zeros(shapes[key[:-6] + "bias"], dtype=torch.bfloat16)
    if hfam.moe == "ernie":
        put(base + ".moe_statics.e_score_correction_bias", sel.view(1, E))
    if hfam.moe == "glm4":
        put(base + ".gate.e_score_correction_bias", sel)
    sname = "shared_experts" if hfam.moe in ("ernie", "glm4") else "shared_expert"
    if f"{base}.{sname}.gate_proj.weight" in shapes:
        if shared is None:
            shared = (z(m.sw, D), z(m.sw, D), z(D, m.sw))
        for n_, t in zip(("gate_proj", "up_proj", "down_proj"), shared):
            put(f"{base}.{sname}.{n_}.weight", t)
            if f"{base}.{sname}.{n_}.bias" in shapes:
                put(f"{base}.{sname}.{n_}.bias", z(shapes[f"{base}.{sname}.{n_}.bias"][0]))
    if base + ".shared_expert_gate.weight" in shapes:
        put(base + ".shared_expert_gate.weight", shared_gate)


# --- Mamba (Mamba-1) -------------------------------------------------------------------------------------------

def _mamba_dims(c):
    return {"hidden": c.hidden_size, "inner": c.intermediate_size, "state": c.state_size, "conv": c.conv_kernel,
            "dt_rank": int(c.time_step_rank), "layers": c.num_hidden_layers}


def _plan_mamba(traw, hraw, tfam, hfam, reasons):
    """Mamba-1 in a bigger Mamba-1 (graph/model_graph.py RC_M1: every dim >=): the residual stream zero-padded
    (RMSNorm eps x d / D, gains x sqrt(d / D)); inner channels, state, dt rank zero-padded -- a zero in_proj row
    gives x = z = 0 and the channel outputs y silu(0) = 0, zero B / C columns add nothing to the state's read-out;
    the conv taps right-aligned in the wider kernel (causal: tap K - 1 is the current step); extra layers the
    identity (out_proj 0)."""
    if tfam.layout != hfam.layout:
        raise Refused(OUT_OF_SCOPE, reasons + [f"mixer: a {tfam.name} target on a {hfam.name} host (a recurrent "
                                               f"mixer needs a host of its own family)"])
    tc, hc = resolved(traw), resolved(hraw)
    T, H = _mamba_dims(tc), _mamba_dims(hc)
    for k in T:
        if H[k] < T[k]:
            reasons.append(f"mamba: host {k} {H[k]} < target {T[k]}")
    if tc.hidden_act != hc.hidden_act:
        reasons.append(f"activation: target {tc.hidden_act!r}, host {hc.hidden_act!r}")
    if tc.use_bias and not hc.use_bias:
        reasons.append("bias: the target's in_proj / out_proj have biases, the host's have none")
    if tc.use_conv_bias and not hc.use_conv_bias:
        reasons.append("bias: the target's conv has a bias, the host's has none")
    if reasons:
        raise Refused(OUT_OF_SCOPE, reasons)
    d, D = T["hidden"], H["hidden"]
    changes = {"vocab_size": tc.vocab_size, "layer_norm_epsilon": tc.layer_norm_epsilon * d / D}
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        changes[k] = getattr(tc, k, None)
    notes = []
    if not tc.tie_word_embeddings and hc.tie_word_embeddings:
        changes["tie_word_embeddings"] = False
        notes.append("the target's LM head is untied and the host ties it: tie_word_embeddings set to false")
    L_t, L_h = T["layers"], H["layers"]
    units = [[(i * L_h // L_t, "both")] for i in range(L_t)]
    hosts = [u[0][0] for u in units]
    lines = [f"hidden   {d} -> {D}: zero-padded; eps {tc.layer_norm_epsilon:g} -> {changes['layer_norm_epsilon']:.6g}, "
             f"norm gains x sqrt({d}/{D})",
             f"layers   {L_t} -> {L_h}: target layers on host layers {_ranges(hosts)}"
             + (f"; the other {L_h - L_t} are the identity (out_proj = 0)" if L_h > L_t else ""),
             f"mixer    inner {T['inner']} -> {H['inner']}, state {T['state']} -> {H['state']}, dt rank "
             f"{T['dt_rank']} -> {H['dt_rank']}: zero-padded; conv kernel {T['conv']} -> {H['conv']} (taps right-aligned)"]
    return Construction(tfam, hfam, d, D, {}, {}, [], "", units, None, None, [], {}, "zero", 1.0, {}, changes, notes,
                        None, {"kind": "mamba", "lines": lines, "T": T, "H": H})


@torch.no_grad()
def _build_mamba(T, tc, hdict, con):
    hc, meta = _meta_host(hdict)
    shapes = {k: tuple(v.shape) for k, v in meta.state_dict().items()}
    tied_h = bool(hc.tie_word_embeddings)
    if tied_h:
        shapes.pop("lm_head.weight", None)
    if tied_h and not T["tied"]:
        raise Refused(OUT_OF_SCOPE, "the host ties its LM head to the embedding and the target does not")
    t = {k: v.to(F64) for k, v in T["sd"].items()}
    Td, Hd = con.extras["T"], con.extras["H"]
    d, D, It, Ih = Td["hidden"], Hd["hidden"], Td["inner"], Hd["inner"]
    Nt, Nh, Kt, Kh, Rt, Rh = Td["state"], Hd["state"], Td["conv"], Hd["conv"], Td["dt_rank"], Hd["dt_rank"]
    gam = math.sqrt(d / D)
    sd = {}

    def put(key, x):
        assert key in shapes and tuple(x.shape) == shapes[key], (key, tuple(x.shape), shapes.get(key))
        sd[key] = _narrow(x.contiguous())

    z = lambda *sh: torch.zeros(*sh, dtype=F64)
    E = z(tc.vocab_size, D)
    E[:, :d] = t["backbone.embeddings.weight"]
    put("backbone.embeddings.weight", E)
    unit_of = {u[0][0]: i for i, u in enumerate(con.units)}
    for j in range(hc.num_hidden_layers):
        hp, i = f"backbone.layers.{j}", unit_of.get(j)
        tp = f"backbone.layers.{i}"
        on = i is not None
        g = z(D)
        if on:
            g[:d] = t[tp + ".norm.weight"] * gam
        put(hp + ".norm.weight", g)
        W = z(2 * Ih, D)
        if on:
            w = t[tp + ".mixer.in_proj.weight"]
            W[:It, :d], W[Ih:Ih + It, :d] = w[:It], w[It:]
        put(hp + ".mixer.in_proj.weight", W)
        if hp + ".mixer.in_proj.bias" in shapes:
            b = z(2 * Ih)
            if on and tp + ".mixer.in_proj.bias" in t:
                bt = t[tp + ".mixer.in_proj.bias"]
                b[:It], b[Ih:Ih + It] = bt[:It], bt[It:]
            put(hp + ".mixer.in_proj.bias", b)
        C = z(Ih, 1, Kh)
        if on:
            C[:It, :, Kh - Kt:] = t[tp + ".mixer.conv1d.weight"]
        put(hp + ".mixer.conv1d.weight", C)
        if hp + ".mixer.conv1d.bias" in shapes:
            b = z(Ih)
            if on and tp + ".mixer.conv1d.bias" in t:
                b[:It] = t[tp + ".mixer.conv1d.bias"]
            put(hp + ".mixer.conv1d.bias", b)
        X = z(Rh + 2 * Nh, Ih)
        if on:
            x = t[tp + ".mixer.x_proj.weight"]
            X[:Rt, :It] = x[:Rt]
            X[Rh:Rh + Nt, :It] = x[Rt:Rt + Nt]
            X[Rh + Nh:Rh + Nh + Nt, :It] = x[Rt + Nt:]
        put(hp + ".mixer.x_proj.weight", X)
        P, pb = z(Ih, Rh), z(Ih)
        if on:
            P[:It, :Rt] = t[tp + ".mixer.dt_proj.weight"]
            pb[:It] = t[tp + ".mixer.dt_proj.bias"]
        put(hp + ".mixer.dt_proj.weight", P)
        put(hp + ".mixer.dt_proj.bias", pb)
        A, Dv = z(Ih, Nh), z(Ih)
        if on:
            A[:It, :Nt] = t[tp + ".mixer.A_log"]
            Dv[:It] = t[tp + ".mixer.D"]
        put(hp + ".mixer.A_log", A)
        put(hp + ".mixer.D", Dv)
        O = z(D, Ih)
        if on:
            O[:d, :It] = t[tp + ".mixer.out_proj.weight"]
        put(hp + ".mixer.out_proj.weight", O)
        if hp + ".mixer.out_proj.bias" in shapes:
            b = z(D)
            if on and tp + ".mixer.out_proj.bias" in t:
                b[:d] = t[tp + ".mixer.out_proj.bias"]
            put(hp + ".mixer.out_proj.bias", b)
    g = z(D)
    g[:d] = t["backbone.norm_f.weight"] * gam
    put("backbone.norm_f.weight", g)
    if not tied_h:
        lm = z(tc.vocab_size, D)
        lm[:, :d] = t["backbone.embeddings.weight"] if T["tied"] else t["lm_head.weight"]
        put("lm_head.weight", lm)
    missing = set(shapes) - set(sd)
    if missing:
        raise RuntimeError(f"host keys not written: {sorted(missing)[:8]}")
    return sd


# --- saving / loading -----------------------------------------------------------------------------------

def copy_generation_config(target, out):
    """The target's generation_config.json (eos / bos / pad ids, stop lists, sampling defaults) into the written
    checkpoint, so that generate() stops and samples as the target does. Without one, the host's would be derived
    from its config, which carries only the target's config-level token ids. Returns the file or None."""
    from transformers import GenerationConfig
    try:
        with warnings_off():
            g = GenerationConfig.from_pretrained(target)
    except (OSError, ValueError):
        return None
    g.save_pretrained(out)
    return out / "generation_config.json"


OUT_MARKER = ".tef-emulate-output"     # written into every output directory; only such a directory is ever deleted


class OutDirError(ValueError):
    """--out names a directory this tool will not overwrite."""


def prepare_out(out):
    """Make `out` an empty output directory and mark it as this tool's. An existing directory is deleted
    only when it is empty or carries OUT_MARKER (an earlier run's output); any other directory, a file, the
    filesystem root, the home directory, the current directory or one of its parents, and a source checkout
    are refused (OutDirError), so a mistyped --out never deletes anything else."""
    out = Path(out).expanduser()
    real = out.resolve()
    cwd = Path.cwd().resolve()
    unsafe = {Path("/").resolve(), Path.home().resolve(), ROOT}
    if real in unsafe or real == cwd or real in cwd.parents or (real / ".git").exists():
        raise OutDirError(f"refusing to use {out} as the output directory (it is the filesystem root, the home "
                          f"directory, the current directory or a parent of it, or a source checkout): "
                          f"choose a new directory")
    if out.exists() or out.is_symlink():
        if not out.is_dir() or out.is_symlink():
            raise OutDirError(f"{out} exists and is not a directory: choose another --out")
        if any(out.iterdir()):
            if not (out / OUT_MARKER).is_file():
                raise OutDirError(f"{out} exists, is not empty and is not an output of this tool (no {OUT_MARKER}"
                                  f"): remove it yourself or choose another --out")
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / OUT_MARKER).write_text("written by the_emulation_frontier (elephantine.emulate / tef_recipes); "
                                  "a later run with the same output directory deletes and rewrites it\n")
    return out


def save_host(out: Path, hdict, sd, tokenizer_src):
    from huggingface_hub import save_torch_state_dict
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(hdict, indent=2) + "\n")
    # experts share one tensor per layer (tied); safetensors needs distinct storage per name
    seen, flat = {}, {}
    for k, v in sd.items():
        flat[k] = v.clone() if v.data_ptr() in seen else v
        seen[v.data_ptr()] = k
    save_torch_state_dict(flat, str(out), max_shard_size="5GB")
    if tokenizer_src is not None:
        tok = _tokenizer(tokenizer_src)
        if tok is not None:                              # (a target repo without tokenizer files: none to copy)
            tok.save_pretrained(out)
        copy_generation_config(tokenizer_src, out)
    return sum((out / f).stat().st_size for f in os.listdir(out) if f.endswith(".safetensors"))


def _tokenizer(src):
    """The repo's tokenizer, or None when it ships none (a config / weights-only test repo)."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(src)
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        return None


def _fixed_ids(out, n):
    """Token-id prompts for a checkpoint without a tokenizer: n fixed sequences inside its vocabulary."""
    V = int(json.loads((Path(out) / "config.json").read_text())["vocab_size"])
    return [torch.tensor([[(97 * i + 31 * j + 1) % V for j in range(8 + 2 * i)]]) for i in range(n)]


def load_model(path, dtype):
    from transformers import AutoModelForCausalLM
    with warnings_off():
        m, info = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, attn_implementation="eager",
                                                       output_loading_info=True)
    bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    if bad:
        raise RuntimeError(f"loading {path}: {bad}")
    if dtype == F64:                    # modules kept in fp32 on load (ERNIE's router): run them at fp64 too
        m = m.to(F64)
    return m.eval()


# --- verification ---------------------------------------------------------------------------------------

def lift_fp32():
    """Context: run a reference with its fp32 casts (Tensor.float, .to(float32), .type, zeros_like(float32),
    softmax(dtype=float32)) kept in fp64, to show the remaining gap is only that rounding."""
    import torch.nn.functional as tF

    @contextlib.contextmanager
    def ctx():
        T, zl, sm = torch.Tensor, torch.zeros_like, tF.softmax
        of, ot, oty = T.float, T.to, T.type
        tF.softmax = lambda v, dim=None, dtype=None, **k: sm(v, dim=dim, dtype=F64 if dtype is torch.float32 else dtype,
                                                            **k)

        def to(self, *a, **k):
            a = tuple(F64 if v is torch.float32 else v for v in a)
            if k.get("dtype") is torch.float32:
                k["dtype"] = F64
            return ot(self, *a, **k)
        T.float, T.to = (lambda self: ot(self, F64)), to
        T.type = lambda self, *a, **k: oty(self, *(F64 if v is torch.float32 else v for v in a), **k)
        osm = T.softmax
        T.softmax = lambda self, *a, dtype=None, **k: osm(self, *a, dtype=F64 if dtype is torch.float32 else dtype, **k)
        torch.zeros_like = lambda v, **k: zl(v, **({**k, "dtype": F64} if k.get("dtype") is torch.float32 else k))
        try:
            yield
        finally:
            T.float, T.to, T.type, T.softmax, torch.zeros_like, tF.softmax = of, ot, oty, osm, zl, sm
    return ctx()


def _lift():
    return lift_fp32()


def _agree(a, b):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"identical": a == b, "first_divergence": None if a == b else n}


@torch.no_grad()
def _run(path, dtype, inputs, new_tokens, lift):
    m = load_model(path, dtype)
    ctx = _lift() if lift else contextlib.nullcontext()
    with ctx:
        logits = [m(input_ids=x).logits.to(F64) for x in inputs]
        greedy = [m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=0)[0, x.shape[1]:].tolist() for x in inputs]
    del m
    return logits, greedy


def verify(target, out, prompts=PROMPTS, new_tokens=NEW_TOKENS, fp64=True, log=print):
    """Target vs the written checkpoint (reloaded from disk): fp32, then fp64 with the fp32 casts lifted."""
    tok = _tokenizer(out)
    inputs = ([tok(p, return_tensors="pt").input_ids for p in prompts] if tok is not None
              else _fixed_ids(out, len(prompts)))           # (no tokenizer: fixed id prompts, ids for text)
    dec = tok.decode if tok is not None else (lambda g: " ".join(map(str, g)))
    res = {"prompts": prompts, "new_tokens": new_tokens}
    for name, dt, lift in [("fp32", torch.float32, False)] + ([("fp64", F64, True)] if fp64 else []):
        t0 = time.time()
        tl, tg = _run(target, dt, inputs, new_tokens, lift)
        hl, hg = _run(out, dt, inputs, new_tokens, lift)
        scale = max(x.abs().max().item() for x in tl)
        diff = max((x - y).abs().max().item() for x, y in zip(tl, hl))
        argmax = sum((x.argmax(-1) == y.argmax(-1)).sum().item() for x, y in zip(tl, hl)) / sum(x.numel() for x in inputs)
        res[name] = {"max_abs_diff": diff, "logit_scale": scale, "rel": diff / scale, "argmax_match": argmax,
                     "greedy": [_agree(a, b) for a, b in zip(hg, tg)],
                     "greedy_text_host": [dec(g) for g in hg],
                     "greedy_text_target": [dec(g) for g in tg], "seconds": round(time.time() - t0, 1)}
        log(f"  {name}: max |d logits| = {diff:.3e}  (logit scale {scale:.1f}, rel {diff / scale:.1e});  "
            f"argmax match {argmax:.0%};  greedy identical {sum(g['identical'] for g in res[name]['greedy'])}"
            f"/{len(prompts)}")
    return res


# --- report ---------------------------------------------------------------------------------------------

def _ranges(xs):
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)


MODE_TEXT = {"tied": "every expert = the dense FFN, router 0", "shared": "dense FFN in the shared expert, routed "
             "experts 0", "dense": "dense layers padded",
             "split": "the FFN dealt over the k experts the selection bias picks, router 0",
             "moe": "router rows copied, each target expert duplicated over its share of the host's top-k"}


def construction_lines(con, tc, hc):
    if con.extras.get("lines"):
        return list(con.extras["lines"])
    t_hd, h_hd = _head_dim(tc), _head_dim(hc)
    L_t, L_h = tc.num_hidden_layers, hc.num_hidden_layers
    hosts = sorted(j for us in con.units for j, _ in us)
    two = any(len(us) > 1 for us in con.units)
    eps = con.config_changes.get(con.hfam.eps_key, con.hfam.eps_fixed)
    lines = [
        f"hidden   {con.d} -> {con.D}: " + (
            "same width" if con.d == con.D and con.style != "mirror" else
            {"zero": "zero-padded", "mean": "padded dims carry the mean (LayerNorm)",
             "mirror": "the [h; -h] mirror (an RMSNorm target on a LayerNorm host)"}[con.style]
            + f"; eps {_eps(tc, con.tfam):g} -> {eps:.6g}" + ("" if con.hfam.eps_key else " (fixed)")
            + ", norm gains x sqrt(" + ("2 " if con.style == "mirror" else "") + f"{con.d}/{con.D})"
            + (" (folded into the readers: the host's norms have no gain)"
               if _norm_kind(hc, con.hfam) == "ln_none" else "")
            + ("; q, k scaled so the qk-norm sees the same factor (one eps serves both)" if _qkn(tc, con.tfam) else ""))
        + ("" if con.a == 1 else f"; residual stream x {con.a:.6g}" + (
            " (the residual multipliers' ratio, so o_proj and down_proj copy as they are)" if con.hfam.eps_key
            else " (the host's eps is fixed)")),
        f"layers   {L_t} -> {L_h}: " + ("target layer i on host layer i" if hosts == list(range(L_t)) and not two else
                                       ("each target layer on two host layers (attention, then FFN: a parallel-block "
                                        "host); " if two else "") + f"target layers on host layers {_ranges(hosts)}")
        + (f"; the other {L_h - len(hosts)} are the identity (their residual writers = 0)" if L_h > len(hosts) else ""),
        (f"heads    MLA: q {tc.num_attention_heads} -> {hc.num_attention_heads} heads (extra heads zeroed), kv latent "
         f"{tc.kv_lora_rank} -> {hc.kv_lora_rank} (each latent dim held {hc.kv_lora_rank // tc.kv_lora_rank} times: "
         f"the hard-coded latent-norm eps sees the same rms), q latent {tc.q_lora_rank} -> {hc.q_lora_rank}, nope "
         f"{tc.qk_nope_head_dim} -> {hc.qk_nope_head_dim}, v {tc.v_head_dim} -> {hc.v_head_dim} zero-padded"
         if con.extras.get("mla") else
         f"heads    q {tc.num_attention_heads} -> {hc.num_attention_heads}, kv {_n_kv(tc)} -> {_n_kv(hc)}: "
        + (f"up to {max(len(v) for v in con.kvmap.values())} target KV heads packed side by side per host KV head "
           f"(blocks of {t_hd} dims)" if con.table and con.table["packed"] else
           f"each target KV head on {len(con.kvmap) // max(_n_kv(tc), 1)} host KV group(s)")
        + f"; {hc.num_attention_heads - len(con.qmap)} query heads and {_n_kv(hc) - len(con.kvmap)} KV heads zeroed"),
        f"head_dim {t_hd} -> {h_hd}: " + (con.rope_note or ("no RoPE (learned positions copied)"
                                                            if con.tfam.rope == "none" else "no change")),
        f"FFN      {_inter(tc, con.tfam)} -> " + ", ".join(sorted({
            (f"{f[1]} (zero-padded units)" if f[0] == "dense" else
             f"MoE: {f[1].E} experts x {f[1].ew}, top-{f[1].k}" + (f" + shared {f[1].sw}" if f[1].sw else ""))
            for f in con.ffn})) +
        ("" if all(m == "dense" for m in con.ffn_mode.values()) else
         "; " + ", ".join(sorted({MODE_TEXT[m] for m in con.ffn_mode.values()}))),
    ]
    if con.moe_m:
        lines.append("experts  " + ", ".join(sorted({f"x {m} copies" for m in con.moe_m.values()})) +
                     " per target expert" + ("; spare experts kept out by the selection bias"
                                             if any(con.ffn[j][1].E > m * 1 for j, m in con.moe_m.items()) else ""))
    if con.host_window is not None:
        lines.append(f"windows  host sliding window set to {con.host_window}")
    return lines


def readme(target, host, plan_text, con, tc, hc, changes, res, size):
    L = [f"# {target} computed by the {hc.model_type} architecture of {host}", ""]
    if con.extras.get("act_lambda"):
        L += ["**LIMIT construction, not exact**: the relation labels this edge a limit (the target's ReLU as "
              "GeLU(lam u) / lam on the host's GeLU, lam = 2^20); the checks below measure how close it is.", ""]
    L += [
         f"Written by `python -m elephantine.emulate {target} {host}` ([the_emulation_frontier]({GITHUB})).",
         f"The config is {host}'s own (`{(hc.architectures or ['?'])[0]}`, stock transformers, no remote code) and the "
         f"weights make it compute {target}: load it with `AutoModelForCausalLM.from_pretrained(this_dir)`. None of "
         f"{host}'s weights are used.", "", "## The edge (graph.edge_plan)", "", "```", plan_text, "```", "",
         "## The construction", ""]
    L += [f"- {x}" for x in construction_lines(con, tc, hc)]
    L += ["", "## Config values changed from the host's own", ""]
    L += [f"- `{k}`: {json.dumps(v) if not isinstance(v, list) or len(v) <= 8 else json.dumps(v[:8])[:-1] + ', ...]'}"
          for k, v in changes.items()]
    L += ["- `vocab_size`: the relation ignores the vocabulary (the embedding and LM head are a resizable front end), "
          "so the host takes the target's", "- `torch_dtype`: float32 (each tensor is stored in the narrowest dtype "
          "that holds it exactly; load with dtype=torch.float64 to run the construction at fp64)"]
    L += [f"- {n}" for n in con.notes]
    L += ["- `generation_config.json`: the target's (its eos / bos / pad ids and generation defaults), so generate() "
          "stops as the target does"]
    L += ["", f"Checkpoint: {size / 1e6:.0f} MB of safetensors.", ""]
    if res:
        L += ["## Verification (the target vs this checkpoint reloaded from disk)", "",
              "| precision | max abs diff of logits | logit scale | argmax match | greedy continuations identical |",
              "|---|---|---|---|---|"]
        for name in ("fp32", "fp64"):
            if name in res:
                r = res[name]
                L.append(f"| {name}{' (fp32 casts lifted)' if name == 'fp64' else ''} | {r['max_abs_diff']:.2e} | "
                         f"{r['logit_scale']:.1f} | {r['argmax_match']:.1%} | "
                         f"{sum(g['identical'] for g in r['greedy'])}/{len(r['greedy'])} |")
        if res.get("fp64_skipped"):
            L.append(f"| fp64 | skipped: {res['fp64_skipped']} | | | |")
        r = res["fp32"]
        L += ["", f"Greedy continuations ({res['new_tokens']} tokens, fp32, host):", ""]
        L += [f"- `{p}` -> `{t!r}`" for p, t in zip(res["prompts"], r["greedy_text_host"])]
    return "\n".join(L) + "\n"


# --- driver ---------------------------------------------------------------------------------------------

def refuse_text(e, plan_text, A=None, B=None, tn="", hn=""):
    lines = [plan_text, ""]
    head = {NOT_AN_EDGE: "REFUSED: not an edge of the headline relation",
            APPROX_EDGE: "REFUSED: the edge is not exact",
            OUT_OF_SCOPE: "REFUSED: an exact edge, but outside the script's scope"}[e.code]
    lines.append(head)
    lines += [f"  - {r}" for r in e.reasons]
    if e.code == NOT_AN_EDGE and A is not None:
        import io
        from contextlib import redirect_stdout
        from graph.construct import construct, pretty
        buf = io.StringIO()
        with redirect_stdout(buf):
            pretty(construct(A, B), tn, hn)
        lines += ["", "graph.construct (every route, either direction):", buf.getvalue().rstrip()]
    return "\n".join(lines)


def emulate(target, host, out, verify_it=True, fp64="auto", prompts=PROMPTS, new_tokens=NEW_TOKENS,
            plan_only=False, log=print, check_relation=True):
    """Build (and verify) the checkpoint. Returns the result dict; raises Refused. check_relation=False
    skips the headline-relation gate (the construction's own checks still apply): synthetic tests only."""
    from graph.edge_plan import format_plan
    traw, hraw = fetch_config(target), fetch_config(host)
    A, B, plan = relation(traw, hraw)
    plan_text = format_plan(plan, target, host)
    try:
        if check_relation:
            plan, con = gate(traw, hraw)
        else:
            con = plan_construction(traw, hraw)
    except Refused as e:
        e.text = refuse_text(e, plan_text, A, B, target, host)
        raise
    tc = resolved(traw)
    hdict = host_config_dict(hraw, con)
    hc = resolved(hdict)
    log(plan_text)
    log("\nconstruction (the script builds this pair):")
    for x in construction_lines(con, tc, hc):
        log("  " + x)
    for n in con.notes:
        log("  note: " + n)
    if plan_only:
        return {"plan": plan, "supported": True}
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    with warnings_off():
        model = AutoModelForCausalLM.from_pretrained(target, dtype=F64, attn_implementation="eager").eval()
    T = read_target(model, tc, con.tfam)
    sd = build_host(T, tc, hdict, con, log)
    del T, model
    out = prepare_out(out)
    size = save_host(out, hdict, sd, target)
    del sd
    log(f"\nwrote {out} ({size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    res = None
    if verify_it:
        auto = fp64 == "auto"
        if auto:
            fp64 = _fits_fp64(hc, tc)
        log("verify (target vs the written checkpoint):")
        res = verify(target, out, prompts, new_tokens, fp64=bool(fp64), log=log)
        if not fp64:
            res["fp64_skipped"] = ("not enough free memory for an fp64 run (--fp64 yes forces it)" if auto
                                   else "--fp64 no")
            log(f"  fp64: skipped ({res['fp64_skipped']})")
    changes = {k: v for k, v in con.config_changes.items() if hraw.get(k) != v}
    result = {"target": target, "host": host, "host_architecture": (hc.architectures or [None])[0],
              "edge": plan, "construction": {
                  "layers": con.units, "layer_pos": con.layer_pos, "q_heads": con.qmap, "kv_heads": con.kvmap,
                  "head_dim_map": con.dmaps, "ffn": con.ffn_mode, "stream": con.style,
                  "expert_copies": con.moe_m, "lines": construction_lines(con, tc, hc)},
              "config_changes": changes, "notes": con.notes, "bytes": size, "verification": res,
              "generation_config": (json.loads((out / "generation_config.json").read_text())
                                    if (out / "generation_config.json").exists() else None)}
    (out / "plan.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    (out / "README.md").write_text(readme(target, host, plan_text, con, tc, hc, changes, res, size))
    return result


def _params(c, fam):
    if fam.layout == "mamba":
        return c.num_hidden_layers * 3 * c.hidden_size * c.intermediate_size + 2 * c.vocab_size * c.hidden_size
    hd, H, KV = _head_dim(c), c.num_attention_heads, _n_kv(c)
    att = c.hidden_size * hd * (2 * H + 2 * KV)
    ffn = 0
    for f in _ffn_layers(c, fam):
        ffn += 3 * c.hidden_size * (f[1] if f[0] == "dense" else f[1].ew * f[1].E + f[1].sw)
    return c.num_hidden_layers * att + ffn + 2 * c.vocab_size * c.hidden_size


def _fits_fp64(hc, tc):
    """fp64 when target + host fit comfortably in the available memory (8 bytes a parameter, x 1.5)."""
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return False
    need = 8 * 1.5 * max(_params(hc, FAMILIES[hc.model_type]), _params(tc, FAMILIES[tc.model_type]))
    return need < 0.8 * avail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", help="Hub repo (or local dir) of the model to emulate")
    ap.add_argument("host", help="Hub repo (or local dir) whose architecture hosts it (only config.json is read)")
    ap.add_argument("--out", help="output directory (required unless --plan-only)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and whether the script builds it; "
                                                             "write nothing")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--fp64", choices=["auto", "yes", "no"], default="auto",
                    help="also verify in fp64 (auto: when it fits in memory)")
    ap.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    args = ap.parse_args(argv)
    if not args.plan_only and not args.out:
        ap.error("--out is required")
    if not args.plan_only:
        try:
            prepare_out(args.out)
        except OutDirError as e:
            ap.error(str(e))
    try:
        emulate(args.target, args.host, args.out, verify_it=not args.no_verify,
                fp64={"auto": "auto", "yes": True, "no": False}[args.fp64], new_tokens=args.new_tokens,
                plan_only=args.plan_only)
    except Refused as e:
        print(getattr(e, "text", str(e)))
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
