"""The zoo page's Variants view: which variants of attention (and of the other components) the snapshot's models
use, and which models use each one.

Every membership is computed per MODEL (repo) from what the classifier reads for that repo's config:
  classify()       the slots (kv, window, pos, qknorm, softcap, sink, crosslayer_kv, gen, mixer, ...) and the
                   FEATURES string (out_gate, vnorm, dsa, nope_layers, bias_qkv, rf_<recurrent family>, ...)
  dims_of()        heads, KV heads, head dim, the MLA latent dims, FFN / expert widths
  zoo_ref.spec_of  the classifier's reading in the card's field names: the layer-kind counts (kc), the per-kind
                   attention shapes from the metrics plan (akinds), K = V, the window, RoPE theta / scaling /
                   rotated fraction, the router score / renorm, the recurrent family (rf)
  the sweep        verify/sweep/findings.json: a few variants the classifier does not read are the code-diff
                   sweep's findings (chunked attention, block-sparse masks, differential attention, position-
                   dependent query temperature, ...). Their membership is the finding's model types (and, where the
                   finding's live rule is a config predicate, the predicate its note states); the tests check the
                   counts against the finding's live_models.
  config keys      a handful of score / embedding scales no slot carries (query_pre_attn_scalar, attention_multiplier,
                   embedding multipliers, ...), each marked with the keys it reads.
A variant whose detection depends on per-type code knowledge is UNKNOWN (not absent) for a model whose type has no
modelling code in the sweep inventory (or whose classifier value is itself unverified: qk-norm "unverified:...",
recurrent family "other_..."). The explanations below are text; nothing in them decides membership.

The view's data block (zoo_variants_data) is embedded in dist/zoo.html by graph/zoo_ref.py; the view itself is
graph/templates/zoo_variants.js (+ zoo_variants.css). Code evidence (file + lines per variant and model type) is
precomputed from local source trees by  python -m graph.zoo_variants --evidence  into graph/data/variant_evidence.json
(text search only: no model code is imported or run). Screenshots: python -m graph.zoo_variants --shots dist/zoo.html
"""
import argparse
import base64
import json
import os
import re
from collections import Counter, defaultdict

from graph import model_graph as mg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVIDENCE = os.path.join(HERE, "data", "variant_evidence.json")
JS = os.path.join(HERE, "templates", "zoo_variants.js")
CSS = os.path.join(HERE, "templates", "zoo_variants.css")
INVENTORY = os.path.join(ROOT, "verify", "sweep", "inventory.json")
FINDINGS = os.path.join(ROOT, "verify", "sweep", "findings.json")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --- one model's facts ---------------------------------------------------------------------------------------

class Facts:
    """What the classifier reads for one repo: slots s, features f, dims, the card spec sp, the LM config d."""
    __slots__ = ("name", "cfg", "d", "mt", "omt", "s", "f", "dims", "sp", "no_code")

    def __init__(self, name, cfg, slots=None, dims=None, spec=None, no_code=False):
        from graph import zoo_ref
        self.name, self.cfg = name, cfg
        self.d = mg._lm_config(cfg)
        self.mt = str(self.d.get("model_type") or "").lower()
        self.omt = str(cfg.get("model_type") or "").lower()
        self.s = slots if slots is not None else mg.classify(cfg)
        self.f = set(filter(None, self.s["feats"].split("+")))
        self.dims = dims if dims is not None else mg.dims_of(cfg)
        self.sp = spec if spec is not None else zoo_ref.spec_of(cfg)
        self.no_code = no_code

    # layer-kind counts (model_graph._attn_counts kc: FU FT LU LT S R)
    def kc(self, *keys):
        k = self.sp.get("kc") or {}
        return sum(int(k.get(x, 0)) for x in keys)

    @property
    def n_full(self):
        return self.kc("FU", "FT")

    @property
    def n_local(self):
        return self.kc("LU", "LT")

    def types(self):
        return {self.mt, self.omt} - {""}

    def key(self, *keys):
        """The first truthy config value among keys (the LM config, then the outer wrapper)."""
        for k in keys:
            for src in (self.d, self.cfg):
                v = src.get(k)
                if v not in (None, False, 0, "", [], {}):
                    return v
        return None


# --- the taxonomy ------------------------------------------------------------------------------------------------
# A variant: id, the component and axis it belongs to, a title, a one-paragraph explanation, a formula / diagram
# hint, the membership predicate over Facts, where membership comes from (src), whether it depends on per-type code
# knowledge (code: unknown without code), the relation rule it maps to (rel), the sweep findings that bear on it
# (find), the code-evidence pattern (ev, a regex searched in the member types' modelling files), and its parent.

COMPONENTS = [
    ("attention", "Attention"), ("norms", "Norms"), ("ffn", "FFN / activation"), ("moe", "MoE routing"),
    ("positions", "Positions"), ("embeddings", "Embeddings / heads"), ("residual", "Residual stream"),
]

AXES = [
    # id, component, title, description
    ("heads", "attention", "Head sharing", "How many key / value heads the query heads read, and whether a layer computes its own K and V."),
    ("mask", "attention", "Mask and pattern", "Which keys each query may attend to, per layer."),
    ("apos", "attention", "Position inside attention", "How position enters the q . k score (the Positions tab has the rest)."),
    ("anorm", "attention", "Normalisation inside attention", "Norms applied to q, k, v or the MLA latents before the dot product."),
    ("score", "attention", "Score modifiers", "Changes to the softmax scores beyond q . k / sqrt(d): caps, sinks, temperatures, scales."),
    ("out", "attention", "Output modifiers", "What happens to the attention output (and the projections' biases)."),
    ("rec", "attention", "Linear / recurrent mixers", "Sequence mixers that are not softmax attention, alone or mixed with it."),
    ("ntype", "norms", "Norm type", "The norm the blocks use."),
    ("nplace", "norms", "Norm placement", "Where the norms sit around each sublayer."),
    ("nother", "norms", "Other norms", "Norms outside the block's two sublayer norms."),
    ("ffn", "ffn", "FFN type", "The MLP of a dense layer."),
    ("act", "ffn", "Activation", "The MLP's activation (the gate activation of a GLU)."),
    ("ffnx", "ffn", "FFN extras", "Clamps and per-layer widths."),
    ("moe", "moe", "Experts", "Mixture of experts: routed experts, shared experts and dense layers."),
    ("mscore", "moe", "Router score", "How router logits become expert weights."),
    ("msel", "moe", "Selection", "How experts are selected and how the selected weights are normalised."),
    ("pos", "positions", "Position scheme", "The positional encoding of the attention layers."),
    ("rscale", "positions", "RoPE scaling", "The RoPE frequency change for long context, as the config's rope_scaling / rope_parameters type."),
    ("rtables", "positions", "RoPE tables", "Several frequency tables, or tables that change with the context."),
    ("head", "embeddings", "Output head", "The LM head: tied or not, capped, and extra prediction heads."),
    ("embed", "embeddings", "Embeddings", "Extra inputs and scales at the embedding."),
    ("resid", "residual", "Residual stream", "How sublayers write into the residual stream, and how many streams there are."),
]


def V(id, axis, title, desc, pred, *, f="", src="classifier", code=False, rel=None, find=(), ev=None, parent=None,
      diagram=None):
    comp = next(c for a, c, _, _ in AXES if a == axis)
    return dict(id=id, comp=comp, axis=axis, title=title, desc=desc, pred=pred, f=f, src=src, code=code, rel=rel,
                find=tuple(find), ev=ev, parent=parent, diagram=diagram)


def _qpas_override(x):
    """query_pre_attn_scalar (Gemma-2 / 3: scale = qpas^-1/2, the config class's default included) != head_dim."""
    q = mg._cfg_value(x.d, "query_pre_attn_scalar")
    hd = x.sp.get("hd")
    return bool(q and hd and abs(float(q) - float(hd)) > 1e-9)


def _attn_mult(x):
    """attention_multiplier (Granite, muP: the softmax scale itself) != 1 / sqrt(head_dim)."""
    a, hd = x.d.get("attention_multiplier"), x.sp.get("hd")
    return bool(isinstance(a, (int, float)) and a and hd and abs(float(a) - hd ** -0.5) > 1e-6)


def _headwise_gate(x):
    if "out_gate" not in x.f:
        return False
    d = x.d
    return bool(d.get("headwise_attn_output_gate") or d.get("use_head_wise_attn_gate")
                or str(d.get("gated_attention_proj_granularity_type") or "").lower() in ("head_wise", "headwise")
                or str(d.get("mla_output_gate_type") or "").lower() in ("head_wise", "headwise")
                or str(d.get("attention_gate_type") or "").lower() in ("head_wise", "headwise")
                or str(d.get("attn_gate") or "").lower() in ("head_wise", "headwise")
                or x.mt in FINDING_TYPES.get("out_gate_headwise", ()))


def _query_temp(x):
    """position_query_temperature (sweep): Qwen-1 use_logn_attn, Llama-4 attn_temperature_tuning (class default on),
    llama_4_scaling_beta in rope_parameters / rope_scaling (Ministral-3, Mistral-4, Nemotron-Labs-Diffusion)."""
    if x.d.get("use_logn_attn"):
        return True
    if x.mt in ("llama4", "llama4_text") and mg._cfg_value(x.d, "attn_temperature_tuning"):
        return True
    for k in ("rope_parameters", "rope_scaling"):
        v = x.d.get(k)
        if isinstance(v, dict) and v.get("llama_4_scaling_beta"):
            return True
    return False


def _dsa_key(x, which):
    d = x.d
    sp = d.get("sparse_config") if isinstance(d.get("sparse_config"), dict) else {}
    if "dsa" not in x.f:
        return False
    if which == "indexer":
        return bool(d.get("index_topk") or d.get("index_n_heads") or d.get("indexer_budget") or d.get("index_top_k"))
    if which == "moba":
        return bool(d.get("moba_topk"))
    if which == "infllm":
        return x.mt == "minicpm_sala" and bool(sp.get("topk"))
    if which == "sparse_cfg":
        return bool((d.get("sparse_attention_config") or {}).get("use_sparse_attention"))
    return False


OUT_GATE_EV = r"attn_output_gate|output_gate|attn_gate|gate_attn|head_wise|headwise|\bg_proj\b|o_gate|gated_attention|attention_gate"
FINDING_TYPES = {}          # finding key -> its model types (lower case), filled from findings.json at import


def _ftype(key):
    return lambda x: bool(x.types() & FINDING_TYPES.get(key, set()))


ATTN_KINDS = {"global": "g", "sliding": "s", "kv-shared": "x"}

_RF_GROUPS = [   # (group id, title, explanation, formula, the classifier's recurrent families in it)
    ("mamba1", "Mamba-1 (selective SSM)",
     "A per-channel state-space recurrence whose decay and input / output maps depend on the token (selective scan). "
     "Variants differ in the normalisation of B, C and dt (Falcon-Mamba, Jamba) and in multi-head layouts.",
     "h_t = exp(dt A) h_{t-1} + dt B x_t;  y_t = C h_t + D x_t", ("mamba1", "mamba1_bcdt", "mamba1_bcdtw", "mamba1_mh")),
    ("mamba2", "Mamba-2 (SSD)",
     "Mamba-2's state-space duality: one scalar decay per head and a gated RMSNorm on the output; PLaMo-2 and "
     "Falcon-H1 (without the norm, or norm before the gate) are variants.",
     "h_t = a_t h_{t-1} + B_t x_t^T (a_t scalar per head);  y = norm(C h) * silu(z)",
     ("mamba2", "mamba2_nonorm", "mamba2_nbg", "plamo2")),
    ("gdn", "Gated DeltaNet",
     "Linear attention with the delta rule: the state is corrected towards v_t along k_t with a learned gate and "
     "decay (Qwen3-Next / Qwen3.5, OLMo-Hybrid with negative eigenvalues, GDN-2, GKA).",
     "S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T;  o_t = S_t q_t", ("gdn", "gdn_negeig", "gdn2", "gka")),
    ("kda", "Kimi Delta Attention",
     "The delta rule with a per-channel (not per-head) decay (Kimi-Linear, GLM-5-Next, Youtu); kda_lb bounds the "
     "decay from below.",
     "S_t = (I - b_t k_t k_t^T) diag(a_t) S_{t-1} + b_t k_t v_t^T", ("kda", "kda_lb")),
    ("rwkv", "RWKV",
     "RWKV's time-mix recurrences (v4 per-channel WKV, v5 / v6 matrix-valued heads, v7 a generalised delta rule).",
     "wkv_t = sum_i exp(-(t-1-i) w + k_i) v_i / sum_i exp(...)", ("rwkv4", "rwkv5", "rwkv6", "rwkv7")),
    ("lightning", "Lightning attention",
     "Linear attention with a fixed per-head decay run in blocks (MiniMax-01 / M1, Ling 2.5 / 2.6, MiniCPM-SALA).",
     "S_t = lambda_h S_{t-1} + k_t^T v_t;  o_t = q_t S_t", ("lightning", "bailing_lightning", "sala_lightning")),
    ("gla", "GLA", "Gated linear attention: a data-dependent per-channel gate on a matrix-valued state.",
     "S_t = diag(a_t) S_{t-1} + k_t^T v_t", ("gla",)),
    ("retnet", "RetNet", "Retention: linear attention with a fixed exponential decay per head and a group norm.",
     "S_t = gamma_h S_{t-1} + k_t^T v_t", ("retnet",)),
    ("rglru", "RG-LRU (Griffin)", "RecurrentGemma's real-gated linear recurrent unit, a diagonal recurrence with input "
     "and recurrence gates.", "h_t = a_t h_{t-1} + sqrt(1 - a_t^2) (i_t x_t)", ("rglru",)),
    ("shortconv", "Gated short convolution", "LFM2's double-gated causal depthwise convolution of a few taps in "
     "place of attention.", "y = C * conv_k(B * x)", ("shortconv",)),
    ("hymba", "Hymba heads", "Attention heads and SSM heads side by side in every layer.",
     "y = norm(attn(x)) + norm(ssm(x))", ("hymba",)),
    ("sambay", "SambaY", "Phi-4-mini-flash's gated memory units and a YOCO-style decoder.", "", ("sambay",)),
    ("jet", "JetBlock", "Jet-Nemotron's linear attention with dynamic convolution kernels.", "", ("jet",)),
]
RF_EV = {"mamba1": r"selective_scan|dt_proj|A_log", "mamba2": r"chunk_scan|ssd|A_log|dt_bias", "gdn": r"delta_rule|GatedDeltaNet",
         "kda": r"delta_rule|KimiDelta|kda", "rwkv": r"wkv|time_mix|rwkv", "lightning": r"lightning|slope|decay",
         "gla": r"\bgla\b|fused_chunk|chunk_gla", "retnet": r"retention", "rglru": r"rg_lru|RGLRU|recurrent_gate",
         "shortconv": r"ShortConv|conv1d|causal_conv", "hymba": r"mamba|ssm", "sambay": r"gmu|GMU|yoco", "jet": r"jet|DynamicShortConvolution"}
RF_TITLES = {"mamba1": "Mamba-1", "mamba1_bcdt": "Mamba-1, normed B / C / dt (Falcon-Mamba)",
             "mamba1_bcdtw": "Mamba-1, normed B / C / dt + weights (Jamba)", "mamba1_mh": "Mamba-1 multi-head (Zamba)",
             "mamba2": "Mamba-2", "mamba2_nonorm": "Mamba-2 without the gated norm", "mamba2_nbg": "Mamba-2, norm before gate",
             "plamo2": "PLaMo-2 Mamba", "gdn": "Gated DeltaNet", "gdn_negeig": "Gated DeltaNet, negative eigenvalues",
             "gdn2": "Gated DeltaNet-2", "gka": "GKA", "kda": "KDA", "kda_lb": "KDA, lower-bounded decay",
             "rwkv4": "RWKV-4", "rwkv5": "RWKV-5", "rwkv6": "RWKV-6", "rwkv7": "RWKV-7", "lightning": "Lightning (MiniMax)",
             "bailing_lightning": "Lightning (Ling 2.5 / 2.6)", "sala_lightning": "Lightning (MiniCPM-SALA)"}

ROPE_SCALING_TITLES = {"yarn": "YaRN", "llama3": "Llama-3 scaling", "linear": "Linear interpolation",
                       "dynamic": "Dynamic NTK", "longrope": "LongRoPE", "su": "LongRoPE (su)", "proportional": "Proportional",
                       "deepseek_yarn": "YaRN (DeepSeek)", "custom": "Custom", "ntk": "NTK"}
ACT_TITLES = {"SiLU": "SiLU / swish (SwiGLU in a GLU)", "GeLU": "GeLU (erf)", "GeLU_tanh": "GeLU (tanh approximation)",
              "relu2": "Squared ReLU", "ReLU": "ReLU", "swigluoai": "SwiGLU-OAI (clamped, gpt-oss)", "mish": "Mish",
              "xielu": "xIELU", "situ": "SiTU", "poly_norm": "PolyNorm"}
SCORE_TITLES = {"softmax": "Softmax", "sigmoid": "Sigmoid", "sqrtsoftplus": "sqrt-softplus", "sparsemixer": "SparseMixer"}
QKN_TITLES = {"head": "Per head (RMSNorm, one gain)", "head_distinct": "Per head, per-head gains", "head_free": "Per head, weight-free",
              "head_post": "Per head, after RoPE", "proj": "Whole projection (RMSNorm)", "proj_ln": "Whole projection (LayerNorm)",
              "head_ln": "Per-head LayerNorm", "head_ln_bias": "Per-head LayerNorm + bias", "ln_shared_bias": "Per-head LayerNorm, shared gain + bias",
              "dsv4": "DeepSeek-V4 (q weight-free, K = V head weighted)", "qkv_input": "Second norm on the attention input (CENO)"}
QKN_DESC = {
    "head": ("RMSNorm over each head's q and k (head_dim wide) with one gain shared by the heads, applied before RoPE "
             "(Qwen3, Gemma-3 / 4, LFM2, GLM-4.5, Step-3.5).",
             "q_h <- g * q_h / rms(q_h);  k_h <- g' * k_h / rms(k_h);  then RoPE"),
    "head_distinct": ("Per-head RMSNorm with a separate gain for every head (PLaMo-2).", "q_h <- g_h * q_h / rms(q_h)"),
    "head_free": ("Per-head RMSNorm with no gain (Llama-4 L2Norm, nanochat): the rms is rotation-invariant, so it may sit "
                  "before or after RoPE.", "q_h <- q_h / rms(q_h)"),
    "head_post": ("Per-head RMSNorm with a gain applied AFTER the rotation (Hunyuan, dhara): no pre-RoPE gain gives the "
                  "same function.", "q_h <- g * R_m q_h / rms(q_h)"),
    "proj": ("One RMSNorm over the entire q projection (and one over k), all heads jointly (OLMo-2 / 3, OLMoE, "
             "MiniMax-M2).", "q <- g * q / rms(q) over n_heads x head_dim"),
    "proj_ln": ("One LayerNorm over the whole projection (OLMo-1 attention_layer_norm, Molmo).", "q <- LN(q) over n_heads x head_dim"),
    "head_ln": ("Per-head LayerNorm with per-head gains and no bias (Cohere, StableLM-2-12B).", "q_h <- g_h * (q_h - mean) / std"),
    "head_ln_bias": ("Per-head LayerNorm with per-head gains and biases (Chameleon).", "q_h <- g_h * (q_h - mean) / std + b_h"),
    "ln_shared_bias": ("Per-head LayerNorm with one gain and bias shared by the heads (Persimmon).", "q_h <- g * (q_h - mean) / std + b"),
    "dsv4": ("DeepSeek-V4: a weight-free per-head norm of q and a weighted norm of the single K = V head.", "q_h <- q_h / rms(q_h);  kv <- g * kv / rms(kv)"),
    "qkv_input": ("CENO's qkv_layernorm: a second RMSNorm on the attention INPUT, a per-token renormalisation before the "
                  "q / k / v projections.", "q, k, v = W norm2(norm1(x))"),
}


def _rf_in(fams):
    def p(x):
        rf = x.sp.get("rf")
        if not rf:
            return False
        if rf.startswith("other_"):
            return None
        return rf in fams
    return p


def _qkn(v):
    def p(x):
        q = x.s["qknorm"]
        if q == "no":
            return None if x.no_code else False
        if q.startswith("unverified"):
            return None
        return q == v
    return p


def _slot(slot, *vals):
    return lambda x: x.s[slot] in vals


def _feat(k):
    return lambda x: k in x.f


def _code(p):
    """A code-dependent predicate: False becomes unknown (None) for a type without modelling code."""
    def q(x):
        r = p(x)
        return None if (r is False and x.no_code) else r
    return q


def _variants():
    out = [
        # --- attention: head sharing
        V("mha", "heads", "MHA", "Multi-head attention: every query head has its own key and value head, so the KV cache "
          "holds n_heads x head_dim per token per layer.", _slot("kv", "MHA"), f="n_kv = n_q", rel=("slot", "kv", "MHA"),
          diagram="heads:mha", ev=r"self\.num_heads|num_attention_heads"),
        V("gqa", "heads", "GQA", "Grouped-query attention: groups of query heads share one key / value head. The cache "
          "shrinks by the group ratio n_q / n_kv (see the GQA ratio histogram).", _slot("kv", "GQA"),
          f="head h reads K, V of group floor(h / (n_q / n_kv))", rel=("slot", "kv", "GQA"), diagram="heads:gqa",
          ev=r"num_key_value_groups|repeat_kv"),
        V("mqa", "heads", "MQA", "Multi-query attention: all query heads share a single key / value head (Falcon-7B, "
          "StarCoder, Gemma-2B).", _slot("kv", "MQA"), f="n_kv = 1", rel=("slot", "kv", "MQA"), diagram="heads:mqa",
          ev=r"multi_query|num_kv_heads|kv_heads"),
        V("mla", "heads", "MLA", "Multi-head latent attention (DeepSeek-V2 / V3, Kimi-K2, GLM-5): keys and values are "
          "up-projected from a small per-token latent c (kv_lora_rank) plus one shared rotary key; the cache holds only "
          "c and the rotary key.", _slot("kv", "MLA"),
          f="c = W_dkv x;  k_h = [W_uk,h c ; RoPE(W_kr x)];  v_h = W_uv,h c", rel=("slot", "kv", "MLA"), diagram="heads:mla",
          ev=r"kv_lora_rank|kv_a_proj"),
        V("kv_none", "heads", "No softmax attention", "A purely recurrent model: no attention layer and no KV cache "
          "(the Linear / recurrent axis says which recurrence).", _slot("kv", "none"), rel=("slot", "kv", "none")),
        V("akinds", "heads", "Per-layer-kind head shapes", "The layer kinds differ in query heads, KV heads or head dim "
          "(Gemma-4: sliding layers 16 KV x 256, global layers 4 KV x 512; MiMo-V2 SWA layers with their own head "
          "counts). Read from the metrics plan, the same reading the KV numbers use; the head-dim and head histograms "
          "filter each kind separately.", lambda x: bool(x.sp.get("akinds")), src="classifier (arch_metrics.plan via spec_of akinds)",
          rel=("dims", "q_cap / kv_cap / head_dim: the widest layer, and per-kind counts through the layer-kind matching"),
          find=("per_layer_head_shapes",), ev=r"global_head_dim|swa_head_dim|num_attention_heads_per_layer|head_dim"),
        V("keqv_global", "heads", "K = V on global layers", "The global (full-window) layers have no v_proj: the value is "
          "computed from the key projection (Gemma-4 attention_k_eq_v). Gemma-4 still caches K and V as two tensors "
          "(V = v_norm(k_proj x), K = RoPE(k_norm(k_proj x))).", lambda x: x.sp.get("keqv") == "full", src="classifier: model_graph._k_eq_v (spec_of keqv = full)",
          f="v = v_norm(W_k x);  k = RoPE(k_norm(W_k x))", rel=("kind", "FT"), ev=r"k_eq_v|attention_k_eq_v"),
        V("keqv_all", "heads", "K = V on every layer", "No layer has an independent value projection: one tensor serves "
          "as key and value (DeepSeek-V4's single K = V head per layer; its cache holds one tensor).",
          lambda x: x.sp.get("keqv") == "all", src="classifier: model_graph._k_eq_v_all (spec_of keqv = all)", f="k = v = W_kv x", rel=("kind", "FT+LT"), ev=r"k_eq_v|kv_proj|wkv"),
        V("v_narrow", "heads", "V narrower than Q / K", "Value heads narrower than query / key heads outside MLA "
          "(MiMo-V2: v_head_dim 128, head_dim 192). The sweep found the classifier reads the Q / K width for V.",
          lambda x: bool(x.types() & FINDING_TYPES.get("v_head_dim_narrow", set()))
          and 0 < mg._num(x.d.get("v_head_dim")) < mg._num(x.d.get("head_dim")),
          src="sweep finding v_head_dim_narrow (types + v_head_dim < head_dim)",
          rel=("finding", "v_head_dim_narrow"), find=("v_head_dim_narrow",), ev=r"v_head_dim"),
        V("cla", "heads", "Cross-layer KV sharing", "Some layers read another layer's keys and values instead of "
          "computing their own (Gemma-3n / 4 E-models' last layers, DeepSeek-V4.1, Zamba-2's shared block, CLA).",
          _slot("crosslayer_kv", "yes"), f="K_l, V_l = K_src(l), V_src(l)", rel=("slot", "crosslayer_kv", "yes"),
          ev=r"kv_shared|num_kv_shared_layers|kv_source|shared_kv|cla"),
        V("cla_tail", "heads", "Trailing layers reuse KV (num_kv_shared_layers)", "The last num_kv_shared_layers layers reuse "
          "the K / V of the last layer of the same type that computed its own (Gemma-3n, Gemma-4 E2B / E4B).",
          lambda x: x.s["crosslayer_kv"] == "yes" and bool(mg._num(x.d.get("num_kv_shared_layers"))), parent="cla",
          rel=("slot", "crosslayer_kv", "yes"), ev=r"num_kv_shared_layers|kv_shared_layer"),
        V("cla_source", "heads", "Explicit KV source layers", "Each layer names the layer whose K / V it reads "
          "(kv_source_layers / a shared-KV index).", lambda x: x.s["crosslayer_kv"] == "yes" and mg._kv_source_shared(x.d)
          and not mg._num(x.d.get("num_kv_shared_layers")), parent="cla", rel=("slot", "crosslayer_kv", "yes"),
          ev=r"kv_source|source_layer"),
        V("cla_factor", "heads", "CLA share factor", "Cross-layer attention: every cla_share_factor consecutive layers "
          "share one K / V (Hunyuan use_cla).", lambda x: x.s["crosslayer_kv"] == "yes" and bool(x.d.get("use_cla"))
          and mg._num(x.d.get("cla_share_factor")) > 1, parent="cla", rel=("slot", "crosslayer_kv", "yes"), ev=r"cla_share_factor"),
        V("yoco", "heads", "YOCO", "You Only Cache Once: the upper half of the stack cross-attends to one KV cache made by "
          "the lower half.", lambda x: x.s["crosslayer_kv"] == "yes" and x.mt == "yoco", parent="cla",
          rel=("slot", "crosslayer_kv", "yes"), ev=r"yoco|cross_decoder"),
        V("cla_block", "heads", "Shared attention block (Zamba-2)", "One attention block shared across many positions "
          "of the stack (Zamba-2's shared transformer block).", lambda x: x.s["crosslayer_kv"] == "yes" and x.mt == "zamba2",
          parent="cla", rel=("slot", "crosslayer_kv", "yes"), ev=r"shared_transformer|hybrid_layer_ids|shared"),
        V("cla_other", "heads", "Other KV sharing", "KV sharing set by another key (cross_layer_kv / a CLA type).",
          lambda x: x.s["crosslayer_kv"] == "yes" and not mg._num(x.d.get("num_kv_shared_layers")) and not mg._kv_source_shared(x.d)
          and not (x.d.get("use_cla") and mg._num(x.d.get("cla_share_factor")) > 1) and x.mt not in ("yoco", "zamba2"),
          parent="cla", rel=("slot", "crosslayer_kv", "yes")),
        # --- attention: mask / pattern
        V("causal_full", "mask", "Full causal", "Every attention layer sees the whole prefix: the causal mask only.",
          lambda x: x.s["kv"] != "none" and x.s["window"] == "full" and x.s["gen"] == "ar", f="j <= i",
          src="classifier: classify() slots window = full, gen = ar, kv != none",
          rel=("slot", "window", "full"), diagram="mask:causal", ev=r"causal_mask|is_causal"),
        V("swa", "mask", "Sliding window", "Some or all attention layers attend only to the last W tokens (the window "
          "histogram gives W).", _slot("window", "local"), f="i - W < j <= i", rel=("slot", "window", "local"),
          diagram="mask:sliding", ev=r"sliding_window", find=("sliding_window_keys_unread",)),
        V("swa_all", "mask", "Sliding on every attention layer", "Every attention layer is windowed (Mistral-7B v0.1, "
          "Phi-3-mini-4k, ...).", lambda x: x.s["window"] == "local" and x.n_full == 0 and x.n_local > 0, parent="swa",
          rel=("slot", "window", "local"), diagram="mask:sliding", ev=r"sliding_window"),
        V("swa_mix", "mask", "Global-local interleave", "Sliding-window layers interleaved with full (global) layers "
          "(Gemma-2: 1:1, Gemma-3 / 4: 5:1, gpt-oss: 1:1). The global:local histogram gives the ratio.",
          lambda x: x.s["window"] == "local" and x.n_full > 0 and x.n_local > 0, parent="swa",
          f="layer l: full if l in G, else sliding", rel=("slot", "window", "local"), diagram="mask:interleave",
          ev=r"layer_types|sliding_window_pattern|sliding_attention"),
        V("swa_unread", "mask", "Sliding, layout not read", "A sliding window whose per-layer layout the classifier could "
          "not read (no layer list or pattern key it knows).", lambda x: x.s["window"] == "local" and x.n_local == 0,
          parent="swa", rel=("slot", "window", "local")),
        V("chunked", "mask", "Chunked attention", "Attention only within fixed blocks of attention_chunk_size tokens, "
          "causal inside a block (Llama-4's RoPE layers, 8192). The classifier reads these layers as full; the sweep "
          "found the chunked mask in the code.", lambda x: bool(x.types() & FINDING_TYPES.get("chunked_attention_unread", set()))
          and bool(x.d.get("attention_chunk_size")), src="sweep finding chunked_attention_unread (types + attention_chunk_size)",
          f="floor(i / C) = floor(j / C), j <= i", rel=("finding", "chunked_attention_unread"),
          find=("chunked_attention_unread",), diagram="mask:chunked", ev=r"chunked_attention|attention_chunk_size"),
        V("blocksparse", "mask", "Block-sparse (fixed)", "A fixed, content-independent block-sparse mask: local blocks plus "
          "strided block columns (Phi-3-small on half its layers).", _ftype("blocksparse_mask"),
          src="sweep finding blocksparse_mask (model types)", rel=("finding", "blocksparse_mask"), find=("blocksparse_mask",),
          diagram="mask:blocksparse", ev=r"blocksparse|vert_stride"),
        V("dsa", "mask", "Top-k sparse attention (DSA and kin)", "A learned scorer keeps the top-k keys (or key blocks) "
          "per query: DeepSeek sparse attention's lightning indexer, MoBA, InfLLM-v2, MiniMax-M3. The relation treats it "
          "as dense: exact while the context is <= index_topk.", _feat("dsa"), f="softmax over the top-k of I(q_i, k_j)",
          rel=("feat", "dsa"), diagram="mask:topk", ev=r"index_topk|indexer|moba|topk"),
        V("dsa_indexer", "mask", "Lightning indexer (DSA)", "DeepSeek-V3.2 / GLM-5 / Mistral-Large-4: an indexer with "
          "index_n_heads small heads scores every key; the top index_topk (2048) are attended.",
          lambda x: _dsa_key(x, "indexer"), parent="dsa", rel=("feat", "dsa"), ev=r"index_topk|Indexer|index_n_heads"),
        V("moba", "mask", "MoBA block top-k", "Mixture of block attention: keys in blocks, the top-k blocks by a pooled "
          "score per query (Ring-mini-sparse).", lambda x: _dsa_key(x, "moba"), parent="dsa", rel=("feat", "dsa"), ev=r"moba"),
        V("infllm", "mask", "InfLLM-v2 block top-k", "MiniCPM-SALA's softmax layers attend to the top-k key blocks.",
          lambda x: _dsa_key(x, "infllm"), parent="dsa", rel=("feat", "dsa"), ev=r"topk|sparse"),
        V("sparse_cfg", "mask", "Sparse attention config (MiniMax-M3)", "A sparse_attention_config with "
          "use_sparse_attention: an index projection selects the keys.", lambda x: _dsa_key(x, "sparse_cfg"), parent="dsa",
          rel=("feat", "dsa"), ev=r"sparse_attention|sparse_index"),
        V("kv_compress", "mask", "Compressed KV pools", "Keys pooled over blocks of 4 / 128 tokens by a softmax-gated "
          "compressor (DeepSeek-V4 CSA / HCA): a pooled key is no single token's key.", _feat("kv_compress"),
          rel=("feat", "kv_compress"), diagram="mask:pooled", ev=r"compress_ratio|compressor|Compressor"),
        V("memory_attn", "mask", "Memory attention", "A second attention over a learned memory of earlier chunks "
          "(EdgeRock, Fabric, CFRD): the sweep found it in the code; the classifier does not read it.",
          _ftype("memory_attention"), src="sweep finding memory_attention (model types)", rel=("finding", "memory_attention"),
          find=("memory_attention",), ev=r"memory"),
        V("prefix_lm", "mask", "Prefix-LM", "Bidirectional over the prompt, causal after it (DIVE-Doc, PaliGemma-style "
          "masks): the sweep found it; the classifier reads the model as causal.",
          lambda x: x.omt == "divedoc" or bool((x.d.get("attn_config") or {}).get("prefix_lm")),
          src="sweep finding prefix_lm_mask (config predicate: model type DIVEdoc / attn_config.prefix_lm)",
          rel=("finding", "prefix_lm_mask"), find=("prefix_lm_mask",), diagram="mask:prefix", ev=r"prefix"),
        V("bidir", "mask", "Bidirectional (diffusion)", "No causal mask: every token sees every token (diffusion LMs "
          "such as LLaDA and Dream, bidirectional embedders).", _slot("gen", "diffusion"), rel=("slot", "gen", "diffusion"),
          diagram="mask:bidir", ev=r"is_causal|bidirectional"),
        V("block_causal", "mask", "Block-causal", "Bidirectional inside a block, causal across blocks (SDAR block "
          "diffusion).", _slot("gen", "block"), f="floor(j / B) <= floor(i / B)", rel=("slot", "gen", "block"),
          diagram="mask:block", ev=r"block_size|is_causal"),
        # --- attention: position inside attention (the Positions tab carries the full axis)
        V("rope", "pos", "RoPE", "Rotary position embedding on the whole head: q and k are rotated pair by pair by "
          "angles proportional to the position, so q . k depends on the distance.", _slot("pos", "rope"),
          f="q_m . k_n = Re[sum_i q_i conj(k_i) e^{i (m - n) theta_i}],  theta_i = base^(-2i / d)",
          rel=("slot", "pos", "rope"), diagram="pos:rope", ev=r"apply_rotary_pos_emb|rotary_emb"),
        V("partial_rope", "pos", "Partial RoPE", "RoPE on only part of each head (rotary_pct / partial_rotary_factor / "
          "rotary_dim); the remaining dimensions carry no position (GPT-NeoX, Phi, GLM, StableLM).",
          _slot("pos", "partial_rope"), f="rotate dims [0, f d); dims [f d, d) unrotated", rel=("slot", "pos", "partial_rope"),
          diagram="pos:partial", ev=r"partial_rotary_factor|rotary_ndims|rotary_dim|rotary_pct"),
        V("nope", "pos", "NoPE", "No positional encoding in the attention: order comes only from the causal mask (and "
          "from any recurrent layers: Jamba, Nemotron-H, Zamba).", _code(_slot("pos", "nope")), code=True,
          rel=("slot", "pos", "nope"), ev=r"position_embedding_type|use_rope|nope"),
        V("nope_layers", "pos", "NoPE layers interleaved", "RoPE on some layers and none on others (SmolLM3, Llama-4, "
          "Flint, Mistral-Large-4).", _feat("nope_layers"), f="layer l: RoPE if l not in N else identity",
          rel=("feat", "nope_layers"), find=("nope_layer_keys_unread",), ev=r"no_rope_layer|use_rope|nope"),
        V("alibi", "pos", "ALiBi", "A per-head linear penalty on the score proportional to the distance (BLOOM, MPT, "
          "Falcon-RW, Baichuan-13B).", _slot("pos", "alibi"), f="score_ij = q_i . k_j / sqrt(d) - m_h (i - j)",
          rel=("slot", "pos", "alibi"), find=("falcon_alibi_slope_scale",), ev=r"alibi"),
        V("learned_abs", "pos", "Learned absolute", "A learned table of position vectors added to the token embeddings "
          "(GPT-2, OPT, GPT-BigCode).", _slot("pos", "learned_abs"), f="x_i = E[t_i] + P[i]",
          rel=("slot", "pos", "learned_abs"), ev=r"wpe|position_embeddings|embed_positions"),
        V("sinusoidal", "pos", "Sinusoidal", "A fixed sine / cosine table added to the embeddings (XGLM, CTRL).",
          _slot("pos", "sinusoidal"), f="P[i, 2k] = sin(i / 10000^(2k/d))", rel=("slot", "pos", "sinusoidal"),
          ev=r"sinusoidal|SinusoidalPositional"),
        V("t5_bias", "pos", "T5 relative bias", "A learned bias per head and log-bucketed distance added to the scores "
          "(T5).", _slot("pos", "t5_relbias"), f="score_ij += b_h[bucket(i - j)]", rel=("slot", "pos", "t5_relbias"),
          ev=r"relative_attention_bias"),
        V("mla_rope", "apos", "Decoupled rotary key (MLA)", "MLA's position path: a separate qk_rope_head_dim-wide rotary "
          "query per head and ONE rotary key shared by the heads, concatenated to the latent (nope) part.",
          lambda x: x.s["kv"] == "MLA" and bool(x.dims.get("mla_rope")), src="classifier: slot kv = MLA and dims_of mla_rope > 0", f="q_h = [q_h^nope ; RoPE(q_h^rope)];  k_h = [k_h^nope ; RoPE(k^rope)]",
          rel=("dims", "mla_rope (the host's shared rotary key must be at least as wide)"), ev=r"qk_rope_head_dim"),
        # --- attention: normalisation
        V("qknorm", "anorm", "qk-norm (any)", "A norm on the queries and keys inside attention, before the dot product. "
          "The flavours below are different functions; the classifier names the one the code builds.",
          _code(lambda x: x.s["qknorm"] != "no"), code=True, src="classifier: classify() slot qknorm != no", rel=("slot", "qknorm", "*"), find=("qknorm_by_code_unlisted", "qknorm_fixed_gain"),
          ev=r"q_norm|k_norm|q_layernorm|query_layernorm"),
    ]
    for v in mg.QKNORM_VARIANTS:
        t, (desc, f) = QKN_TITLES.get(v, v), QKN_DESC.get(v, (v, ""))
        out.append(V("qkn_" + v, "anorm", "qk-norm: " + t[0].lower() + t[1:], desc, _qkn(v), f=f, code=True, parent="qknorm",
                     rel=("slot", "qknorm", v), ev=r"q_norm|k_norm|q_layernorm|qk_norm|query_layernorm"))
    out += [
        V("vnorm", "anorm", "V norm", "A per-head RMS norm of the values (Gemma-4 v_norm, weight-free; DeepSeek-V4).",
          _feat("vnorm"), f="v_h <- v_h / rms(v_h)", rel=("feat", "vnorm"), ev=r"\bv_norm\b|value_norm|\bkv_norm\b"),
        V("mla_kv_norm", "anorm", "MLA latent norm (kv)", "The RMSNorm on MLA's compressed kv latent before the up-"
          "projections (kv_a_layernorm): in every MLA code, it cannot be switched off.", _slot("kv", "MLA"),
          src="classifier: classify() slot kv = MLA (every MLA code builds kv_a_layernorm)",
          f="c = RMSNorm(W_dkv x)", rel=("dims", "MLA_FOLD_PAYABLE = False: folding the latent norm is measured not payable"),
          ev=r"kv_a_layernorm"),
        V("mla_q_norm", "anorm", "MLA latent norm (q)", "MLA with a low-rank query (q_lora_rank) also normalises the "
          "query latent (q_a_layernorm).", lambda x: x.s["kv"] == "MLA" and bool(x.dims.get("q_latent")), src="classifier: slot kv = MLA and dims_of q_latent > 0",
          f="c_q = RMSNorm(W_dq x)", rel=("dims", "q_latent (present in both or neither)"), ev=r"q_a_layernorm"),
        V("clip_qkv", "anorm", "Clipped q / k / v", "q, k, v clamped to [-c, c] (OLMo-1 clip_qkv, MPT).",
          _feat("clip_qkv"), f="q <- clamp(q, -c, c)", rel=("feat", "clip_qkv"), ev=r"clip_qkv"),
        # --- attention: score modifiers
        V("softcap_attn", "score", "Attention-logit softcap", "c tanh(s / c) on the attention scores, bounding them by "
          "c (Gemma-2: c = 50).", _slot("softcap", "attn"), f="s_ij <- c tanh(s_ij / c)", rel=("slot", "softcap", "attn"),
          ev=r"attn_logit_softcapping|softcap=|softcap:|if softcap"),
        V("sink_learned", "score", "Learned attention sinks", "A learned per-head logit in the softmax denominator, so a "
          "head can attend to nothing (gpt-oss, MiMo-V2's sliding layers, Flint, DeepSeek-V4).",
          _code(_slot("sink", "yes")), code=True, f="p_ij = e^{s_ij} / (e^{b_h} + sum_j e^{s_ij})",
          rel=("slot", "sink", "yes"), ev=r"sinks|s_aux|sink"),
        V("sink_tokens", "score", "Start-token sinks", "The first few tokens stay visible to every query of a sliding "
          "window (StreamingLLM-style kept tokens: PseudoBrain pb_attention_sinks).",
          lambda x: bool(x.key("pb_attention_sinks", "num_sink_tokens", "attention_sink_size", "sink_tokens")),
          src="config keys pb_attention_sinks / num_sink_tokens / attention_sink_size (no classifier slot)",
          rel=("none", "no slot or feature: the classifier reads the window only"), diagram="mask:sinktok",
          ev=r"sink"),
        V("scale_qpas", "score", "Scale override: query_pre_attn_scalar", "Gemma-2 / 3 scale the scores by "
          "query_pre_attn_scalar^-1/2 instead of head_dim^-1/2; listed where the two differ (Gemma-2-27B: 144 vs 128).",
          _qpas_override, src="config query_pre_attn_scalar (and the config class's default) vs the classifier's head_dim",
          f="s = q . k / sqrt(qpas)", rel=("none", "a scale: a free config value (folds into W_q)"),
          ev=r"query_pre_attn_scalar"),
        V("scale_mult", "score", "Scale override: attention_multiplier", "The softmax scale given directly (Granite muP "
          "attention_multiplier), listed where it differs from head_dim^-1/2.", _attn_mult,
          src="config attention_multiplier vs the classifier's head_dim", f="s = a (q . k)",
          rel=("none", "a scale: a free config value (folds into W_q)"), ev=r"attention_multiplier"),
        V("scale_qk_gain", "score", "Fixed gain after qk-norm", "q and k multiplied by a fixed qk_scale after a weight-free "
          "norm (nanochat ports: 1.2), a sharper softmax than 1 / sqrt(d). The sweep found it in the code.",
          lambda x: bool(x.types() & FINDING_TYPES.get("qknorm_fixed_gain", set())) and bool(x.key("qk_scale", "qk_sharpen_scale")),
          src="sweep finding qknorm_fixed_gain (types + qk_scale)", f="q, k <- c q / rms(q), c k / rms(k)",
          rel=("finding", "qknorm_fixed_gain"), find=("qknorm_fixed_gain",), ev=r"qk_scale|sharpen"),
        V("scale_layer", "score", "Per-layer score scaling", "Scores divided by the layer index + 1 (GPT-2 "
          "scale_attn_by_inverse_layer_idx).", lambda x: bool(x.d.get("scale_attn_by_inverse_layer_idx")),
          src="config scale_attn_by_inverse_layer_idx", f="s /= (l + 1)", rel=("none", "not read by the relation"),
          ev=r"scale_attn_by_inverse_layer_idx"),
        V("qtemp", "score", "Position-dependent query temperature", "q scaled by a factor that grows with the position "
          "inside the context (Qwen-1 logn, Llama-4 attn_temperature_tuning on NoPE layers, llama_4_scaling_beta in "
          "Ministral-3 / Mistral-4). The relation caps the credited context at the threshold.", _query_temp,
          src="sweep finding position_query_temperature (its config predicate)", f="q_i <- q_i (1 + beta log(1 + floor(i / L0)))",
          rel=("finding", "position_query_temperature"), find=("position_query_temperature",),
          ev=r"logn|attn_temperature|llama_4_scaling_beta|attn_scale"),
        V("diff_attn", "score", "Differential attention", "Two softmax maps per head, one subtracted from the other with "
          "a learned lambda, cancelling common-mode attention (Motif-3's GDLA). The sweep found it in the code.",
          _ftype("diff_attn"), src="sweep finding diff_attn (model types)", f="o = softmax(s1) V - lambda softmax(s2) V",
          rel=("finding", "diff_attn"), find=("diff_attn",), ev=r"lambda|noise"),
        # --- attention: output modifiers
        V("out_gate", "out", "Output gate", "The attention output multiplied by a data-dependent sigmoid gate before "
          "o_proj (Qwen3-Next / 3.5, Laguna, Step-3.5, Kimi MLA, ...).", _code(_feat("out_gate")), code=True,
          f="o <- o * sigmoid(W_g x)", rel=("feat", "out_gate"), find=("out_gate_by_code_unlisted", "out_gate_activation"),
          ev=OUT_GATE_EV),
        V("out_gate_head", "out", "Head-wise output gate", "One scalar gate per head (Laguna, Spark-2.5, Youtu, Ling-3, "
          "Step-3.5).", _headwise_gate, parent="out_gate", code=True,
          src="classifier out_gate + the head-wise keys / sweep finding out_gate_headwise", f="o_h <- o_h * sigmoid(w_h . x)",
          rel=("feat", "out_gate"), find=("out_gate_headwise",), ev=OUT_GATE_EV),
        V("out_gate_elem", "out", "Element-wise output gate", "A gate per channel of every head (Qwen3-Next / 3.5 "
          "attn_output_gate).", lambda x: "out_gate" in x.f and not _headwise_gate(x), parent="out_gate",
          f="o <- o * sigmoid(W_g x)  (n_heads x head_dim gates)", rel=("feat", "out_gate"), ev=OUT_GATE_EV),
        V("bias_qkv", "out", "q / k / v biases", "Bias vectors on the q, k, v projections (Qwen-1 / 2 / 2.5, GPT-2, "
          "OPT, Bloom, Pythia, Phi).", _code(_feat("bias_qkv")), code=True, f="q = W_q x + b_q", rel=("feat", "bias_qkv"),
          find=("bias_qkv_by_code_unlisted", "bias_qkv_table_wrong"), ev=r"bias=True|attention_bias|qkv_bias"),
        V("bias_out", "out", "Output-side biases", "Biases on o_proj and the MLP, and affine LayerNorm betas (the "
          "classifier's bias_out feature covers all three).", _code(_feat("bias_out")), code=True, f="y = W_o o + b_o",
          rel=("feat", "bias_out"), find=("bias_out_by_code_unlisted",), ev=r"bias=True|mlp_bias|use_bias"),
        # --- attention: recurrent
        V("hybrid", "rec", "Hybrid (softmax + recurrent)", "Softmax-attention layers interleaved with recurrent ones "
          "(the attention-share histogram gives the mix).", _slot("mixer", "hybrid"), rel=("slot", "mixer", "hybrid"),
          ev=r"layers_block_type|layer_types|linear_attention"),
        V("pure_rec", "rec", "Recurrent only", "No softmax attention at all: an SSM or linear-attention stack (Mamba, "
          "RWKV, RetNet, GLA).", _slot("mixer", "ssm", "linear"), rel=("slot", "mixer", "ssm"),
          src="classifier: classify() slot mixer in (ssm, linear)"),
        V("par_heads", "rec", "Attention and SSM in one layer", "An attention branch and an SSM branch side by side in "
          "the same layer (Falcon-H1, Hymba, PseudoBrain hybrid layers).",
          lambda x: any(c in (x.sp.get("sched") or "") for c in "PQ") or (x.mt in ("falcon_h1", "hymba") and x.s["mixer"] == "hybrid"),
          src="classifier (spec_of schedule: parallel attention + recurrent layers)", rel=("slot", "mixer", "hybrid"),
          ev=r"mamba|ssm"),
    ]
    for gid, title, desc, f, fams in _RF_GROUPS:
        out.append(V("rf_" + gid, "rec", title, desc, _rf_in(set(fams)), f=f, rel=("feat", "rf_" + fams[0]),
                     ev=RF_EV.get(gid), src="classifier: recurrent family (_recurrent_family, spec_of rf) in " + ", ".join(fams)))
        if len(fams) > 1:
            for fam in fams:
                out.append(V("rf_" + gid + ":" + fam, "rec", RF_TITLES.get(fam, fam), f"The classifier's recurrent family "
                             f"{fam} (model_graph.RECURRENT_FAMILY / _recurrent_family).", _rf_in({fam}),
                             parent="rf_" + gid, rel=("feat", "rf_" + fam),
                             src=f"classifier: recurrent family (_recurrent_family, spec_of rf) = {fam}"))
    out += [
        # --- norms
        V("rmsnorm", "ntype", "RMSNorm", "x / rms(x) times a learned per-channel gain: no mean subtraction, no bias.",
          _slot("norm", "RMSNorm"), f="y = g * x / sqrt(mean(x^2) + eps)", rel=("slot", "norm", "RMSNorm"),
          ev=r"RMSNorm"),
        V("layernorm", "ntype", "LayerNorm", "Mean subtraction, division by the standard deviation, a learned gain and "
          "bias (GPT-2, BERT, Falcon, Phi-2).", _slot("norm", "LayerNorm"), f="y = g * (x - mu) / sigma + b",
          rel=("slot", "norm", "LayerNorm"), ev=r"LayerNorm"),
        V("pre_norm", "nplace", "Pre-norm", "Each sublayer reads norm(x); its output is added to the un-normalised "
          "residual. The default of modern decoders.",
          lambda x: (None if x.no_code else True) if not (x.f & {"post_norm", "post_only", "post_ln"}) else False,
          code=True, f="x + f(norm(x))", rel=("feat", "post_norm", "absent"),
          src="classifier: none of the features post_norm, post_only, post_ln", find=("norm_placement_keys_unread",),
          ev=r"input_layernorm|ln_1"),
        V("sandwich", "nplace", "Sandwich norm", "A pre-norm and a second norm on each sublayer's output before the "
          "residual add (Gemma-2 / 3 / 4, GLM-4-0414, AFMoE, LFM3).", _code(_feat("post_norm")), code=True,
          f="x + norm2(f(norm1(x)))", rel=("feat", "post_norm"), find=("norm_placement_keys_unread",),
          ev=r"post_attention_layernorm|post_feedforward_layernorm|post_norm"),
        V("post_only", "nplace", "Post-only norm", "No pre-norm: each sublayer reads the raw residual and its output is "
          "normalised before the add (OLMo-2 / 3, EXAONE-4).", _code(_feat("post_only")), code=True, f="x + norm(f(x))",
          rel=("feat", "post_only"), ev=r"post_attention_layernorm|post_feedforward_layernorm"),
        V("post_ln", "nplace", "Post-LN", "The original Transformer block: the residual stream itself is normalised "
          "after every add (GPT-1, OPT-350m).", _feat("post_ln"), f="norm(x + f(x))", rel=("feat", "post_ln"),
          ev=r"do_layer_norm_before|post"),
        V("emb_norm", "nother", "Embedding norm", "A norm on the embedding output before the first block (BLOOM "
          "word_embeddings_layernorm; weight-free RMSNorm in nanochat, gaggle, Muse-Glimmer). The sweep found it; the "
          "classifier does not read it.",
          lambda x: bool(x.types() & (FINDING_TYPES.get("embedding_norm", set()) - {"mpt"})),
          src="sweep finding embedding_norm (its types; MPT only with an embedding norm, none in the snapshot)",
          rel=("finding", "embedding_norm"), find=("embedding_norm",), ev=r"word_embeddings_layernorm|embed_norm|rms_norm\(|norm\("),
        V("selfcond_norm", "nother", "Self-conditioning norm", "An always-on weight-free RMSNorm at the end of a "
          "self-conditioning path (DiffusionGemma).", _feat("selfcond_norm"), rel=("feat", "selfcond_norm"),
          ev=r"self_condition"),
        V("grouped_norm", "nother", "Grouped RMSNorm", "An RMSNorm over channel groups (layernorm_num_groups: K2-Horizon, "
          "Jais-ASR). The sweep found it; the classifier does not read it.",
          lambda x: bool(x.types() & FINDING_TYPES.get("grouped_rmsnorm", set())) and mg._num(x.d.get("layernorm_num_groups")) > 1,
          src="sweep finding grouped_rmsnorm (types + layernorm_num_groups > 1)", rel=("finding", "grouped_rmsnorm"),
          find=("grouped_rmsnorm",), ev=r"num_groups|group"),
        # --- FFN
        V("glu", "ffn", "Gated MLP (GLU)", "down(act(gate x) * up x): SwiGLU with SiLU, GeGLU with GeLU.",
          _slot("ffn", "glu"), f="y = W_d (act(W_g x) * W_u x)", rel=("slot", "ffn", "glu"),
          ev=r"gate_proj|w1|w3"),
        V("vanilla", "ffn", "Plain MLP", "down(act(up x)), no gate (GPT-2, OPT, Falcon, Phi-2, Nemotron-H).",
          _slot("ffn", "vanilla"), f="y = W_d act(W_u x)", rel=("slot", "ffn", "vanilla"),
          ev=r"c_fc|fc1|dense_h_to_4h|up_proj"),
        V("moe_any", "ffn", "Mixture of experts", "A router picks the top-k of E expert MLPs per token (the MoE "
          "routing tab breaks it down).", _slot("ffn", "moe", "moe_shared"), rel=("slot", "ffn", "moe"),
          src="classifier: classify() slot ffn in (moe, moe_shared)",
          ev=r"experts|router|gate"),
        V("hadamard", "ffn", "Hadamard MLP", "Cactus needle's HadamardMLP: no gate / up / down projections; its width "
          "is the next power of two of the hidden size.",
          _slot("ffn", "hadamard"), rel=("slot", "ffn", "hadamard"), ev=r"hadamard|Hadamard"),
        V("ffn_none", "ffn", "No MLP", "Blocks with no FFN: the mixer is the whole block (Mamba-1, Falcon-Mamba, "
          "Mamba-2).", _slot("ffn", "none"), rel=("slot", "ffn", "none"), ev=r"mixer|MambaBlock"),
        V("act_clip", "ffnx", "Clamped SwiGLU", "A SwiGLU with clamps on gate and up (DeepSeek-V4 swiglu_limit, "
          "Step-3.5, Hy4); gpt-oss's own clamped form is the SwiGLU-OAI activation.", _feat("act_clip"),
          f="silu(min(g, L)) * clamp(u, -L, L)", rel=("feat", "act_clip"), ev=r"swiglu_limit|clamp"),
        V("ffn_widths", "ffnx", "Per-layer FFN widths", "FFN widths that change with depth (OpenELM's ffn_multipliers, "
          "Nemotron-NAS).", lambda x: bool(x.sp.get("ffn_widths")) and x.sp["ffn_widths"][0] != x.sp["ffn_widths"][1],
          src="classifier (dims_of ffn_widths)", rel=("dims", "ffn widths: each target layer needs a host layer at least as wide"),
          ev=r"ffn_multipliers|intermediate_size"),
        # --- MoE
        V("moe", "moe", "Routed experts only", "MoE layers with routed experts and no always-on shared expert "
          "(Mixtral, gpt-oss, Qwen3-MoE).", _slot("ffn", "moe"), rel=("slot", "ffn", "moe"), ev=r"experts|num_experts"),
        V("moe_shared", "moe", "Routed + shared experts", "MoE layers with one or more always-on shared experts "
          "(DeepSeek-V2 / V3, Qwen2-MoE, Llama-4, Kimi-K2).", _slot("ffn", "moe_shared"), f="y = sum_topk w_e E_e(x) + S(x)",
          rel=("slot", "ffn", "moe_shared"), ev=r"shared_expert"),
        V("shared_gate", "moe", "Gated shared expert", "A sigmoid gate on the shared expert (Qwen1.5 / 2-MoE, Qwen3-Next, "
          "Qwen3.5-MoE).", _feat("shared_gate"), f="y += sigmoid(w . x) S(x)", rel=("feat", "shared_gate"),
          find=("shared_gate_undetected",), ev=r"shared_expert_gate"),
        V("moe_dense", "moe", "Dense layers in a MoE", "Some layers keep a plain MLP (DeepSeek-V3's first 3, "
          "first_k_dense_replace, mlp_only_layers, Llama-4 interleaving).", lambda x: bool(x.sp.get("ndense")),
          src="classifier (dims_of moe_sig n_dense)", rel=("dims", "n_moe / n_dense layer counts"), ev=r"first_k_dense|mlp_only_layers|dense"),
        V("moe_latent", "moe", "Latent MoE", "Experts run in a down-projected latent (Kimi-K3 routed_expert_hidden_size, "
          "Nemotron-3).", lambda x: bool(x.sp.get("moe_latent")), src="classifier (dims_of moe_sig latent)",
          rel=("dims", "moe latent width"), ev=r"latent"),
        V("router_bias", "msel", "Selection-only expert bias", "Aux-loss-free balancing: experts are selected on l + b "
          "but weighted by l (DeepSeek-V3 noaux_tc, e_score_correction_bias).", _feat("router_bias"),
          f="E_sel = topk(l + b);  w_e = s(l_e)", rel=("feat", "router_bias"), find=("router_bias_unread",),
          ev=r"e_score_correction_bias|expert_bias|noaux"),
        V("router_group", "msel", "Group-limited routing", "Experts split into n_group groups; each token routes only "
          "inside its topk_group best groups (DeepSeek-V3).", _feat("router_group"), rel=("feat", "router_group"),
          find=("group_mask_zero_fill",), ev=r"n_group|topk_group"),
        V("router_in_gate", "msel", "Input-gated router", "The router score scales the expert INPUT (Llama-4).",
          _feat("router_in_gate"), f="E(x * sigmoid(l))", rel=("feat", "router_in_gate"), ev=r"sigmoid"),
        V("hash_route", "msel", "Hash routing", "A token-id table fixes the experts of the first layers (DeepSeek-V4).",
          _feat("hash_route"), rel=("feat", "hash_route"), ev=r"hash|tid2eid"),
        # --- positions (scaling / tables)
        V("rope_scaled", "rscale", "RoPE scaling (any)", "A change to the RoPE frequencies for long context: the "
          "config's rope_scaling / rope_parameters type.", lambda x: bool(x.sp.get("scaling")),
          src="classifier reading (zoo_ref._rope: rope_scaling / rope_parameters type)",
          rel=("none", "RoPE scaling is a free config value; the rot gate prices the tables"), ev=r"rope_type|rope_scaling"),
        V("rope_tables", "rtables", "Per-layer-type RoPE tables", "Separate RoPE tables for sliding and global layers "
          "(Gemma-3 rope_local_base_freq, rope_parameters per layer type, MiMo swa_rope_theta).",
          lambda x: bool(x.sp.get("theta_local")), src="classifier reading (zoo_ref._rope theta_local)",
          rel=("dims", "rope_sched / rope_cap: the rot gate (rope_fits)"), find=("local_rope_theta_unread",),
          ev=r"rope_local_base_freq|sliding_attention|rope_parameters|local_rope"),
        V("rope_tables_code", "rtables", "Per-layer-type RoPE tables (code only)", "A second table on sliding layers "
          "that the classifier does not see (OLMo-3: YaRN on full layers only; PLaMo-3 rope_local_theta): the sweep "
          "found it in the code.", _ftype("local_rope_theta_unread"), src="sweep finding local_rope_theta_unread (model types)",
          rel=("finding", "local_rope_theta_unread"), find=("local_rope_theta_unread",), ev=r"rope_local|sliding|rope_parameters"),
        V("dyn_ntk", "rtables", "Context-dependent RoPE base", "Dynamic NTK that rebuilds the RoPE base as the sequence "
          "grows inside the context (Qwen-1 use_dynamic_ntk, Meditor's global layers).",
          lambda x: bool(x.types() & FINDING_TYPES.get("dynamic_ntk_in_context", set()))
          and (bool(x.d.get("use_dynamic_ntk")) or x.mt.startswith("meditor")),
          src="sweep finding dynamic_ntk_in_context (its config predicate)", rel=("finding", "dynamic_ntk_in_context"),
          find=("dynamic_ntk_in_context",), ev=r"ntk"),
        # --- embeddings / heads
        V("tied", "head", "Tied embeddings", "The output head is the transposed input embedding table.",
          _feat("tied_emb"), f="lm_head = E^T", rel=("feat", "tied_emb"), find=("tie_default_unread",),
          ev=r"tie_word_embeddings|_tied_weights_keys"),
        V("untied", "head", "Untied head", "A separate lm_head matrix.", lambda x: "tied_emb" not in x.f,
          rel=("feat", "tied_emb", "absent"), ev=r"lm_head"),
        V("softcap_final", "head", "Final-logit softcap", "c tanh(z / c) on the output logits (Gemma-2 / 3n / 4, "
          "nanochat, RecurrentGemma).", lambda x: x.s["softcap"] in ("attn", "final") and bool(x.sp.get("cap_final")),
          f="z <- c tanh(z / c)", rel=("slot", "softcap", "final"),
          src="classifier: classify() slot softcap in (attn, final) with a final cap (spec_of cap_final)", ev=r"final_logit_softcapping|logit_softcap"),
        V("mtp", "head", "Multi-token prediction", "Extra heads trained to predict tokens further ahead (DeepSeek-V3, "
          "GLM-4.5, MiMo); they do not change the next-token logits.", _slot("mtp", "yes"), rel=("slot", "mtp", "yes"),
          ev=r"nextn|mtp"),
        V("logit_mult", "head", "Logit multiplier", "A fixed scale on the output logits (Granite logits_scaling, "
          "Falcon-H1 lm_head_multiplier).", lambda x: bool(x.key("logits_scaling", "lm_head_multiplier", "output_multiplier", "logit_scale")),
          src="config keys logits_scaling / lm_head_multiplier / output_multiplier / logit_scale",
          rel=("none", "a scale: a free config value"), ev=r"logits_scaling|lm_head_multiplier|logit_scale"),
        V("ple", "embed", "Per-layer embeddings", "A second token embedding fed into every layer (Gemma-3n / Gemma-4 "
          "E-models).", _feat("ple"), rel=("feat", "ple"), find=("qwen4_ple_reads_ngrams",), ev=r"per_layer_input|embed_tokens_per_layer"),
        V("value_emb", "embed", "Value embeddings / residuals", "Extra per-token inputs added inside attention "
          "(nanochat value embeddings, value residuals).", _feat("value_emb"), rel=("feat", "value_emb"),
          find=("value_emb_keys_unread",), ev=r"value_emb|ve_gate|value_residual"),
        V("engram", "embed", "Engram n-gram memory", "A hashed n-gram memory added to the residual (DeepSeek-V4.1).",
          _feat("engram"), rel=("feat", "engram"), find=("ngram_memory_unread",), ev=r"engram"),
        V("emb_mult", "embed", "Embedding multiplier", "A fixed scale on the input embeddings (Granite / muP "
          "embedding_multiplier, MiniCPM scale_emb, Falcon-H1, RecurrentGemma).",
          lambda x: bool(x.key("embedding_multiplier", "scale_emb", "embedding_scale", "mup_embedding_multiplier",
                               "embeddings_scale_by_sqrt_dim", "scale_embedding")),
          src="config keys embedding_multiplier / scale_emb / embedding_scale / mup_embedding_multiplier / "
              "embeddings_scale_by_sqrt_dim / scale_embedding", rel=("none", "a scale: a free config value"),
          ev=r"embedding_multiplier|scale_emb|embed_scale"),
        V("selfcond", "embed", "Self-conditioning", "The previous step's logits fed back as an input (DiffusionGemma, "
          "LLaDA2.1-ORL).", _feat("selfcond"), rel=("feat", "selfcond"), ev=r"self_condition"),
        V("switch_lora", "embed", "Switched LoRA", "Per-token LoRA adapters switched on a control token "
          "(Granite-Switch).", _feat("switch_lora"), rel=("feat", "switch_lora"), ev=r"adapter|lora"),
        # --- residual
        V("sequential", "resid", "Sequential block", "The MLP reads the stream after attention has been added.",
          _slot("block", "sequential"), f="x' = x + attn(n(x));  x'' = x' + mlp(n(x'))",
          rel=("slot", "block", "sequential"), ev=r"residual"),
        V("parallel", "resid", "Parallel block", "Attention and MLP read the same input and their outputs are summed "
          "(GPT-J, Pythia, Falcon, Cohere).", _slot("block", "parallel"), f="x + attn(n(x)) + mlp(n(x))",
          rel=("slot", "block", "parallel"), find=("parallel_by_code_unlisted",), ev=r"parallel"),
        V("hyper_conn", "resid", "Hyper-connections", "Several residual streams mixed by learned (Sinkhorn) maps "
          "(DeepSeek-V4, GLM-5-Next).", _feat("hyper_conn"), f="X_{l+1} = H_res X_l + H_post f(H_pre X_l)",
          rel=("feat", "hyper_conn"), find=("hyper_conn_variant",), ev=r"hc_mult|sinkhorn|hyper"),
        V("attn_res", "resid", "Attention residuals", "The residual is a softmax-weighted mix of earlier block outputs "
          "(Kimi AttnRes).", _feat("attn_res"), rel=("feat", "attn_res"), find=("attn_res_keys_unread",), ev=r"attn_res"),
        V("looped", "resid", "Looped depth", "The same layers run several times (recurrent depth).", _slot("loop", "looped"),
          rel=("slot", "loop", "looped"), find=("loop_depth_unread",), ev=r"loop"),
        V("resid_mult", "resid", "Residual multiplier", "A fixed scale on each sublayer's write into the residual "
          "(Granite residual_multiplier, MiniCPM scale_depth).", lambda x: bool(x.key("residual_multiplier", "scale_depth")),
          src="config keys residual_multiplier / scale_depth", rel=("none", "a scale: a free config value"),
          ev=r"residual_multiplier|scale_depth"),
        V("zamba_shared", "resid", "Zamba shared block", "Zamba / Zamba2's shared transformer block on concat(x, "
          "embedding).", _feat("zamba_shared"), rel=("feat", "zamba_shared"), ev=r"shared_transformer|concat"),
    ]
    # the attention position axis repeats the Positions tab's scheme variants (one object, two tabs)
    for v in out:
        if v["axis"] == "pos" and v["id"] not in ("mla_rope",):
            v["also"] = "apos"
    return out


def _dynamic_variants(facts):
    """Variants whose values are read from the data: RoPE scaling types, activations, router scores, renorm."""
    out = []
    seen = Counter(x.sp.get("scaling") for x in facts if x.sp.get("scaling"))
    for t, _ in seen.most_common():
        out.append(V("rope_scale:" + t, "rscale", ROPE_SCALING_TITLES.get(t, t), f"rope_scaling / rope_parameters type "
                     f"'{t}' (zoo_ref._rope).", (lambda t: lambda x: x.sp.get("scaling") == t)(t), parent="rope_scaled",
                     src="classifier reading (zoo_ref._rope)", rel=("none", "RoPE scaling is a free config value"),
                     ev=r"rope_type|rope_scaling|" + re.escape(t)))
    acts = Counter(x.s["act"] for x in facts)
    for a, _ in acts.most_common():
        out.append(V("act:" + a, "act", ACT_TITLES.get(a, a), f"The activation the classifier reads for the MLP: {a} "
                     f"(raw_act: the family's code table, the config key, else the config class default).",
                     (lambda a: lambda x: x.s["act"] == a)(a), rel=("slot", "act", a),
                     ev=r"hidden_act|ACT2FN|act_fn"))
    scores = Counter(x.sp.get("score") for x in facts if x.sp.get("score"))
    for sc, _ in scores.most_common():
        out.append(V("score:" + sc, "mscore", SCORE_TITLES.get(sc, sc), f"Router score '{sc}' (dims_of moe_sig).",
                     (lambda sc: lambda x: x.sp.get("score") == sc)(sc), src="classifier (dims_of moe_sig score)",
                     rel=("dims", "moe_why: the router score must match (a code-fixed or config value per family)"),
                     ev=r"scoring_func|sigmoid|softmax"))
    rn = Counter(str(x.sp.get("renorm")) for x in facts if x.sp.get("renorm") not in (None,))
    title = {"R": "Top-k renormalised", "P": "Top-k not renormalised"}
    for r, _ in rn.most_common():
        out.append(V("renorm:" + r, "msel", title.get(r, "Renorm: " + r), "Whether the selected experts' weights are "
                     "renormalised to sum to 1 (norm_topk_prob / the family's code).",
                     (lambda r: lambda x: str(x.sp.get("renorm")) == r)(r), src="classifier (dims_of moe_sig renorm)",
                     rel=("none", "norm_topk_prob is a free config value"), ev=r"norm_topk_prob|renorm"))
    return out


# --- numeric facets ------------------------------------------------------------------------------------------------
# id, component, title, per-layer-kind?, log bins?, extractor -> value or {kind: value}; None = not applicable,
# "?" = applicable but unknown

def _per_kind(x, field):
    """{kind: value} over the model's attention layer kinds: from the metrics plan's per-kind shapes when they differ
    (akinds), else the one shape on every kind present (kc). g global / full window, s sliding, x KV-shared."""
    if x.s["kv"] == "none":
        return None
    ak = x.sp.get("akinds")
    out = {}
    if ak:
        for e in ak:
            k = ATTN_KINDS[e["k"]]
            if k in out:
                continue
            v = {"hd": e.get("hd"), "h": e.get("h"), "kvh": e.get("kvh")}.get(field) if field != "ratio" else (
                e["h"] / e["kvh"] if e.get("h") and e.get("kvh") else None)
            out[k] = v if v else "?"
        return out
    sp = x.sp
    v = {"hd": sp.get("hd"), "h": sp.get("h"), "kvh": sp.get("kvh") or sp.get("h"),
         "ratio": (sp["h"] / (sp.get("kvh") or sp["h"])) if sp.get("h") else None}[field]
    v = v if v else "?"
    kinds = []
    if x.n_full:
        kinds.append("g")
    if x.n_local:
        kinds.append("s")
    if x.kc("S"):
        kinds.append("x")
    if not kinds:
        kinds = ["s"] if x.s["window"] == "local" else ["g"]
    return {k: v for k in kinds}


def _theta(x):
    if x.s["pos"] not in ("rope", "partial_rope"):
        return None
    t, tl = x.sp.get("theta"), x.sp.get("theta_local")
    out = {}
    if x.n_full or not x.n_local:
        out["g"] = t or "?"
    if x.n_local or x.s["window"] == "local":
        out["s"] = tl or t or "?"
    return out


def _rot(x):
    if x.s["pos"] == "rope":
        return 1.0
    if x.s["pos"] != "partial_rope":
        return None
    r = x.sp.get("rot")
    if r is None and x.sp.get("rot_dim") and x.sp.get("hd"):
        r = x.sp["rot_dim"] / x.sp["hd"]
    return round(r, 4) if r else "?"


def _gl(x):
    if x.s["window"] != "local" or not (x.n_full and x.n_local):
        return None
    return round(x.n_local / x.n_full, 3)


def _attn_share(x):
    if x.s["mixer"] != "hybrid":
        return None
    a, r = x.kc("FU", "FT", "LU", "LT", "S"), x.kc("R")
    return round(a / (a + r), 4) if a + r else "?"


def _mla(field):
    return lambda x: ((x.sp.get("mla") or {}).get(field) or "?") if x.s["kv"] == "MLA" else None


def _moe(field):
    def f(x):
        if x.s["ffn"] not in ("moe", "moe_shared"):
            return None
        v = x.sp.get(field)
        if field in ("nsh", "ndense", "shw"):
            return v or 0
        return v or "?"
    return f


def _dense_ffn(x):
    if x.s["ffn"] in ("moe", "moe_shared"):
        return None
    return x.sp.get("ffn") or "?"


def _ffn_ratio(x):
    v = _dense_ffn(x)
    if v in (None, "?") or not x.sp.get("d"):
        return v if v is None else "?"
    return round(v / x.sp["d"], 3)


FACETS = [
    # id, component, title, per-kind, log, extractor, unit / note
    ("hd", "attention", "Head dim", True, False, lambda x: _per_kind(x, "hd"), "q / k width per head"),
    ("h", "attention", "Query heads", True, False, lambda x: _per_kind(x, "h"), ""),
    ("kvh", "attention", "KV heads", True, False, lambda x: _per_kind(x, "kvh"), "MLA: one per query head"),
    ("ratio", "attention", "GQA ratio (q / kv heads)", True, False, lambda x: _per_kind(x, "ratio"), ""),
    ("win", "attention", "Sliding window", False, True,
     lambda x: (x.sp.get("win") or "?") if x.s["window"] == "local" else None, "tokens"),
    ("gl", "attention", "Local layers per global layer", False, False, _gl, "sliding : full"),
    ("ashare", "attention", "Attention share of a hybrid", False, False, _attn_share, "attention layers / mixer layers"),
    ("mla_kv", "attention", "MLA kv latent rank", False, False, _mla("kv"), "kv_lora_rank"),
    ("mla_q", "attention", "MLA q latent rank", False, False, _mla("q"), "q_lora_rank (0: full-rank query)"),
    ("mla_rope", "attention", "MLA rope head dim", False, False, _mla("rope"), "qk_rope_head_dim"),
    ("mla_nope", "attention", "MLA nope head dim", False, False, _mla("nope"), "qk_nope_head_dim"),
    ("mla_v", "attention", "MLA v head dim", False, False, _mla("v"), "v_head_dim"),
    ("theta", "positions", "RoPE theta", True, True, _theta, "the base; sliding layers' own base where kept"),
    ("rot", "positions", "Rotated fraction", False, False, _rot, "1 = full RoPE"),
    ("ctx", "positions", "Max context", False, True, lambda x: x.sp.get("ctx") or "?", "max_position_embeddings"),
    ("d", "ffn", "d_model", False, True, lambda x: x.sp.get("d") or "?", ""),
    ("L", "ffn", "Layers", False, False, lambda x: x.sp.get("Le") or x.sp.get("L") or "?", "unrolled depth"),
    ("ffn", "ffn", "FFN width (dense)", False, True, _dense_ffn, "intermediate_size"),
    ("ffn_ratio", "ffn", "FFN ratio (width / d_model)", False, False, _ffn_ratio, ""),
    ("E", "moe", "Routed experts", False, True, _moe("E"), ""),
    ("k", "moe", "Experts per token (top-k)", False, False, _moe("k"), ""),
    ("ew", "moe", "Expert width", False, True, _moe("ew"), ""),
    ("nsh", "moe", "Shared experts", False, False, _moe("nsh"), "0: none"),
    ("ndense", "moe", "Dense layers", False, False, _moe("ndense"), ""),
    ("vocab", "embeddings", "Vocab", False, True, lambda x: x.sp.get("vocab") or "?", "embedding rows"),
]
KINDS = ("g", "s", "x")
# facets whose values are few and meaningful one by one (head dims, head counts, ranks): one bar per value
DISCRETE_FACETS = {"hd", "h", "kvh", "ratio", "gl", "mla_kv", "mla_q", "mla_rope", "mla_nope", "mla_v", "rot", "k", "nsh",
                   "ndense", "L", "E"}


# --- the relation's view of a variant ---------------------------------------------------------------------------

def relation(rel, observed):
    """What the relation does with a variant, as a host and as a target, from its rules:
    ("slot", slot, value)    the slot order (payable_reaches / edge_cost) against every other value the slot takes
                             in the snapshot (observed[slot]); capacity is checked separately
    ("feat", key[, absent])  FEATURES[key] (target side, host side, why); rf_ families default (wall, free)
    ("kind", kinds)          the layer-kind matching (KIND_REACH)
    ("dims", text)           a capacity axis
    ("finding", key)         not modelled: a sweep finding
    ("none", text)           not read by the relation"""
    if not rel:
        return None
    kind = rel[0]
    if kind == "slot":
        slot, v = rel[1], rel[2]
        order = dict(mg.SLOTS)[slot]
        vals = sorted(observed.get(slot, ()))
        if v == "*":
            return {"k": "slot", "slot": slot, "note": "each flavour below is its own slot value; see the flavours"}
        tgt, host = [], []
        for u in vals:
            if u == v:
                continue
            if mg.payable_reaches(slot, v, u):
                c = mg.edge_cost(order, v, u)
                tgt.append([u, "+".join(c[0]) if c else "", bool(c and c[1])])
            if mg.payable_reaches(slot, u, v):
                c = mg.edge_cost(order, u, v)
                host.append([u, "+".join(c[0]) if c else "", bool(c and c[1])])
        zero = [[t, note] for (s, t, h), note in mg.ZERO_EDGES.items() if s == slot and h == v]
        return {"k": "slot", "slot": slot, "v": v, "t": tgt, "h": host, "of": [u for u in vals if u != v], "zero": zero}
    if kind == "feat":
        key = rel[1]
        rule = mg.FEATURES.get(key) or (("wall", "free", "a recurrent family: a host needs the same family (rf_ "
                                         "features are walls as targets, free as hosts)") if key.startswith("rf_")
                                        else ("free", "free", ""))
        out = {"k": "feat", "feat": key, "t": rule[0], "h": rule[1], "why": rule[2]}
        if len(rel) > 2:
            out["absent"] = True
        return out
    if kind == "kind":
        return {"k": "kind", "kinds": rel[1], "reach": {k: list(v) for k, v in mg.KIND_REACH.items()}}
    if kind in ("dims", "none"):
        return {"k": kind, "note": rel[1]}
    if kind == "finding":
        return {"k": "finding", "key": rel[1]}
    return None


# --- the data block ------------------------------------------------------------------------------------------------

def _bits(flags):
    """A membership list (one bool per model) as base64 bytes, bit 7 of byte 0 = model 0."""
    b = bytearray((len(flags) + 7) // 8)
    for i, x in enumerate(flags):
        if x:
            b[i >> 3] |= 0x80 >> (i & 7)
    return base64.b64encode(bytes(b)).decode()


def no_code_types(inv):
    """Model types the sweep found no modelling code for (inventory no_code), lower case."""
    if not inv:
        return set()
    return {t.lower() for t, r in (inv.get("types") or {}).items() if r.get("no_code")}


def model_facts(nodes, configs, inv=None):
    """Facts for every repo of every page node, in page order (node by node, members in order)."""
    nc = no_code_types(inv)
    out = []
    for nd in nodes:
        for m in nd["members"]:
            cfg = configs[m]
            f = Facts(m, cfg)
            f.no_code = bool(f.types() & nc) or (inv is not None and not (f.types() & {t.lower() for t in inv.get("types", {})}))
            out.append(f)
    return out


def set_findings(findings):
    FINDING_TYPES.clear()
    for f in findings or []:
        FINDING_TYPES[f["key"]] = {t.lower() for t in f.get("types", [])}


def all_variants(facts):
    return _variants() + _dynamic_variants(facts)


def membership(variants, facts):
    """variant id -> list of True / False / None (unknown) per model."""
    return {v["id"]: [v["pred"](x) for x in facts] for v in variants}


def zoo_variants_data(nodes, configs, meta=None, inv=None, findings=None, fw=None, hub=None, evidence=None):
    """The Variants view's data block, aligned with the page's nodes (models enumerated node by node, members in
    order). meta: graph/model_meta.json; hub: repo -> downloads; fw: frameworks.json; evidence: variant_evidence.json."""
    set_findings(findings)
    meta = meta or {}
    facts = model_facts(nodes, configs, inv)
    variants = all_variants(facts)
    mem = membership(variants, facts)
    n = len(facts)
    observed = defaultdict(set)
    for x in facts:
        for k, _ in mg.SLOTS:
            observed[k].add(x.s[k])
    node_of = []
    for i, nd in enumerate(nodes):
        node_of += [i] * len(nd["members"])
    dates = [((meta.get(x.name) or {}).get("created_at") or "")[:10] for x in facts]
    types = sorted({x.mt for x in facts})
    tix = {t: k for k, t in enumerate(types)}
    fnames = list(((fw or {}).get("frameworks") or {}).keys())
    frepo = (fw or {}).get("repos") or {}

    def fwchar(r, f):
        e = frepo.get(r) or {}
        nat, vt = e.get("native") or {}, e.get("via_transformers") or {}
        return "1" if nat.get(f) is True else "t" if vt.get(f) else "0" if nat.get(f) is False else "?"
    fwm = ["".join(fwchar(x.name, f) for f in fnames) for x in facts]
    fkeys = {f["key"]: k for k, f in enumerate(findings or [])}
    VS = []
    for v in variants:
        flags = mem[v["id"]]
        yes = [i for i, r in enumerate(flags) if r]
        unk = [i for i, r in enumerate(flags) if r is None]
        dated = sorted((dates[i], facts[i].name) for i in yes if dates[i])
        fwc = {}
        for k, f in enumerate(fnames):
            known = [fwm[i][k] for i in yes if fwm[i][k] != "?"]
            if known:
                fwc[f] = [known.count("1"), known.count("t"), len(known)]
        src = v["src"]
        if src == "classifier" and v["parent"]:
            src = f"classifier: {v['parent']} split by the config keys / values named here"
        elif src == "classifier" and v["rel"] and v["rel"][0] == "slot" and v["rel"][2] != "*":
            src = f"classifier: classify() slot {v['rel'][1]} = {v['rel'][2]}"
        elif src == "classifier" and v["rel"] and v["rel"][0] == "feat":
            src = f"classifier: classify() feature {v['rel'][1]}" + (" absent" if len(v["rel"]) > 2 else "")
        rec = {"id": v["id"], "c": v["comp"], "a": v["axis"], "t": v["title"], "d": v["desc"], "f": v["f"],
               "src": src, "m": _bits(flags), "n": len(yes), "u": len(unk),
               "s": len({node_of[i] for i in yes}),
               "first": list(dated[0]) if dated else None,
               "rel": relation(v["rel"], observed), "find": [fkeys[k] for k in v["find"] if k in fkeys],
               "fw": fwc}
        if unk:
            rec["um"] = _bits([r is None for r in flags])
        if v["parent"]:
            rec["p"] = v["parent"]
        if v.get("also"):
            rec["also"] = v["also"]
        if v["code"]:
            rec["code"] = 1
        if v["diagram"]:
            rec["dg"] = v["diagram"]
        if evidence and evidence.get("variants", {}).get(v["id"]):
            rec["ev"] = evidence["variants"][v["id"]]
        VS.append(rec)
    FS = []
    for fid, comp, title, perkind, log, fn, unit in FACETS:
        vals = [fn(x) for x in facts]
        rec = {"id": fid, "c": comp, "t": title, "unit": unit, "log": int(log), "pk": int(perkind),
               "disc": int(fid in DISCRETE_FACETS)}
        if perkind:
            rec["v"] = {k: [(v or {}).get(k) if isinstance(v, dict) else None for v in vals] for k in KINDS}
            rec["v"] = {k: a for k, a in rec["v"].items() if any(z is not None for z in a)}
        else:
            rec["v"] = vals
        FS.append(rec)
    fl = [{"key": f["key"], "note": f.get("note", ""), "dir": f.get("direction", ""), "n": f.get("live_models", 0),
           "rule": f.get("live_rule", ""), "types": f.get("types", []), "ex": f.get("example_live", [])[:5],
           "suggest": (f.get("suggest") or [""])[0]} for f in (findings or [])]
    used = sorted({k for v in VS for k in v["find"]})
    return {
        "n": n, "comps": [{"id": c, "t": t} for c, t in COMPONENTS],
        "axes": [{"id": a, "c": c, "t": t, "d": d} for a, c, t, d in AXES],
        "variants": VS, "facets": FS,
        "models": {"dl": [int((hub or {}).get(x.name) or 0) for x in facts], "date": dates,
                   "p": [int((meta.get(x.name) or {}).get("hf_total") or 0) for x in facts],
                   "mt": [tix[x.mt] for x in facts], "nc": _bits([x.no_code for x in facts])},
        "types": types, "fw_names": fnames, "fw": fwm,
        "findings": {str(k): fl[k] for k in used},
        "evidence": {k: evidence[k] for k in ("source", "version") if k in evidence} if evidence else None,
    }


def inject(html, nodes, configs, meta=None, inv=None, findings=None, fw=None, hub=None, evidence_path=EVIDENCE):
    """Fill the template's Variants placeholders (data, JS, CSS); a template without them is returned as is."""
    if "__ZOO_VARIANTS__" not in html:
        return html
    data = zoo_variants_data(nodes, configs, meta=meta, inv=inv, findings=findings, fw=fw, hub=hub,
                             evidence=_load(evidence_path))
    enc = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    js = open(JS).read().replace("</script", "<\\/script") if os.path.exists(JS) else ""
    css = open(CSS).read() if os.path.exists(CSS) else ""
    return html.replace("__ZOO_VARIANTS__", enc).replace("__ZOO_VARIANTS_JS__", js).replace("__ZOO_VARIANTS_CSS__", css)


# --- code evidence: file + lines per variant and model type ----------------------------------------------------------

def _tf_version(src):
    try:
        txt = open(os.path.join(src, "__init__.py")).read()
        m = re.search(r'__version__ = "([^"]+)"', txt)
        return m.group(1) if m else None
    except OSError:
        return None


def _type_files(inv, remote_index, tf_src):
    """model type (lower) -> [(kind, local path, url-ready ref)] modelling files, modeling_*.py first."""
    out = {}
    for t, rec in (inv.get("types") or {}).items():
        files = []
        for c in rec.get("code", []):
            v = c.get("variant", "")
            for f in c.get("files", []):
                if v.startswith("tf5:") and f.startswith("tf5:") and tf_src:
                    p = os.path.join(tf_src, f[4:])
                    if os.path.exists(p):
                        files.append(("tf", p, f[4:]))
                elif v.startswith("hub:"):
                    repo, _, fn = f.rpartition("/")
                    lp = ((remote_index or {}).get(repo) or {}).get(fn)
                    if lp and os.path.exists(lp):
                        m = re.search(r"/snapshots/([0-9a-f]+)/", lp)
                        files.append(("hub", lp, (repo, m.group(1) if m else "main", fn)))
        files.sort(key=lambda e: (e[0] != "tf", "modeling_" not in os.path.basename(e[1]), "configuration" in e[1]))
        out[t.lower()] = files
    return out


def build_evidence(nodes, configs, inv, findings, tf_src, remote_index=None, per_variant=4, per_file=3):
    """variant id -> [{type, n, file, url, lines: [[line, text]]}]: for the variant's most common member types, the
    first lines of the type's modelling file matching the variant's pattern (a text search of local files)."""
    set_findings(findings)
    facts = model_facts(nodes, configs, inv)
    variants = all_variants(facts)
    files = _type_files(inv, remote_index, tf_src)
    ver = _tf_version(tf_src) if tf_src else None
    cache = {}
    out = {}
    for v in variants:
        if not v["ev"]:
            continue
        rx = re.compile(v["ev"])
        cnt = Counter(x.mt for x in facts if v["pred"](x))
        recs = []
        for t, k in cnt.most_common():
            if len(recs) >= per_variant:
                break
            for kind, path, ref in files.get(t, []):
                if path not in cache:
                    try:
                        cache[path] = open(path, errors="replace").read().splitlines()
                    except OSError:
                        cache[path] = []
                hits = [[i + 1, ln.strip()[:160]] for i, ln in enumerate(cache[path])
                        if rx.search(ln) and not ln.strip().startswith(("#", "from ", "import "))][:per_file]
                if not hits:
                    continue
                if kind == "tf":
                    base = f"https://github.com/huggingface/transformers/blob/v{ver}/src/transformers/{ref}" if ver else None
                    rec = {"type": t, "n": k, "file": "transformers/" + ref, "url": base}
                else:
                    repo, sha, fn = ref
                    rec = {"type": t, "n": k, "file": f"{repo}/{fn}", "url": f"https://huggingface.co/{repo}/blob/{sha}/{fn}"}
                rec["lines"] = hits
                recs.append(rec)
                break
        if recs:
            out[v["id"]] = recs
    return {"source": "transformers " + (ver or "?") + " (local source tree) and the Hub files in the sweep's remote "
                      "index, searched as text", "version": ver, "variants": out}


SHOTS = [  # name, hash, viewport, dark
    ("variants_01_attention", "#v=variants", (1400, 1000), False),
    ("variants_02_detail_sinks", "#v=variants&zp=detail&zv=sink_learned", (1400, 1000), False),
    ("variants_03_combined_filter", "#v=variants&zi=gqa,swa,qkn_head&zn=hd~any~128~128~0&zfrom=2025-01", (1400, 1000), False),
    ("variants_04_cooc_attention_x_moe", "#v=variants&zp=cooc&zca=attention&zcb=moe", (1400, 1000), False),
    ("variants_05_timeline_heads", "#v=variants&zp=timeline", (1400, 1000), False),
    ("variants_06_positions_dims", "#v=variants&zc=positions", (1400, 1000), False),
    ("variants_07_moe_detail_router_bias", "#v=variants&zc=moe&zp=detail&zv=router_bias", (1400, 1000), False),
    ("variants_08_models_carry_mla_dsa", "#v=models&zi=mla,dsa&zm=1", (1400, 1000), False),
    ("variants_09_detail_keqv_dark", "#v=variants&zp=detail&zv=keqv_global", (1400, 1000), True),
    ("variants_10_phone", "#v=variants&zi=sink_learned", (375, 800), False),
]


def shots(page, out):
    """Screenshots of the Variants view (headless Chromium, file:// permalinks) into out/variants_*.png."""
    from playwright.sync_api import sync_playwright
    os.makedirs(out, exist_ok=True)
    url = "file://" + os.path.abspath(page)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, h, (w, ht), dark in SHOTS:
            phone = w < 640
            ctx = b.new_context(viewport={"width": w, "height": ht}, color_scheme="dark" if dark else "light",
                                is_mobile=phone, has_touch=phone, device_scale_factor=2 if phone else 1)
            pg = ctx.new_page()
            errs = []
            pg.on("pageerror", lambda e, errs=errs: errs.append(str(e)))
            pg.goto(url + h)
            pg.wait_for_selector("#stats .stat")
            target = "#view-models" if "v=models" in h else "#view-variants"
            pg.wait_for_selector(target + ":not([hidden])")
            pg.evaluate("""s => { const e = document.querySelector(s), c = document.querySelector('.controls');
                window.scrollTo(0, e.getBoundingClientRect().top + window.scrollY - (c && getComputedStyle(c).position === 'sticky' ? c.offsetHeight + 8 : 8)); }""", target)
            pg.wait_for_timeout(300)
            pg.screenshot(path=os.path.join(out, name + ".png"))
            ctx.close()
            assert not errs, (name, errs)
            print("wrote", os.path.join(out, name + ".png"))
        b.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evidence", action="store_true", help="write graph/data/variant_evidence.json")
    ap.add_argument("--shots", metavar="PAGE", help="screenshot the Variants view of PAGE (e.g. dist/zoo.html) into --shots-out")
    ap.add_argument("--shots-out", default="dist/zoo_shots")
    ap.add_argument("--tf-src", default=os.environ.get("TF5_SRC", ""),
                    help="a transformers 5 source tree (the package directory), searched as text")
    ap.add_argument("--remote-index", default=os.path.join(os.environ.get("TEF_SCRATCH", os.path.expanduser("~/.cache/the_emulation_frontier")), "code_sweep", "remote_index.json"),
                    help="the sweep's index of downloaded Hub files (repo -> file -> local path)")
    ap.add_argument("--out", default=EVIDENCE)
    a = ap.parse_args()
    if a.shots:
        shots(a.shots, a.shots_out)
        return
    if not a.evidence:
        ap.print_help()
        return
    from graph import zoo_ref
    from graph.model_graph import ZOO_CONFIGS, _dedup_points
    items = list(mg.iter_json_configs(ZOO_CONFIGS))
    configs = dict(items)
    models = []
    for name, d in items:
        if mg._junky(name):
            continue
        c = mg.classify(d)
        if c is not None:
            models.append({"name": name, "slots": c, "dims": mg.dims_of(d)})
    P = _dedup_points(models)
    rank = {m["name"]: k for k, m in enumerate(models)}
    nodes = [{"members": sorted(p["members"], key=lambda m: rank.get(m, len(rank)))} for p in P]
    ev = build_evidence(nodes, configs, zoo_ref._load_json(INVENTORY), zoo_ref._load_json(FINDINGS), a.tf_src,
                        _load(a.remote_index))
    with open(a.out, "w") as f:
        json.dump(ev, f, indent=1, sort_keys=True)
    print(f"wrote {a.out}: {len(ev['variants'])} variants with evidence ({ev['source']})")


if __name__ == "__main__":
    main()
