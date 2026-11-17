"""Build the model subset graph from real HF configs.

For every cached model we read its config, classify each architectural slot
(norm / positional / KV-sharing / FFN / activation), group models with identical
slot-tuples into architecture CLASSES, then draw an edge  A -> B  whenever every
slot of A is <= the matching slot of B in that slot's partial order (A is a
special case of B; B can be parameterized to reproduce A). We transitively
reduce to covering edges and render with graphviz `dot`.

Edge currency = the union of the differing slots' costs; an edge is EXACT unless
a differing slot uses an approximate (magnitude-limit) emulation, in which case
it is drawn dashed.

Run:
  python -m graph.model_graph                 # all cached models
  python -m graph.model_graph --out graph/out # write .dot/.svg/.png there
"""
from __future__ import annotations

import argparse
import os
import subprocess

from transformers import AutoConfig

# --- round 3 (verify/r3/specs): every change is keyed by its spec name ------------------------------------------------
# R3_OFF switches a spec's change off (the leave-one-out tables, verify/r3/relcheck.py --loo). A table entry a spec adds
# or drops is an _R3Set / _R3Dict entry keyed by the spec, so the use sites read the tables unchanged; a code change is
# guarded by `if _r3("<spec>")`.
# Round 3, batch 1: the tables' OWN storage is kept equal to the live content (R3_OFF's add / discard / ... resync every
# table), because CPython's set / dict fast paths read a subclass's storage directly and bypass __iter__ /
# __contains__: set(x), frozenset(x), `plain_set & x`, `plain_set | x`, {**x}, dict(x), x.isdisjoint(...). A rebinding
# of R3_OFF to a plain set is caught at the next method call (_sync compares the frozen key).
_R3_TABLES = []


class _R3Off(set):
    """R3_OFF: a set whose mutations resync every round-3 table."""

    def _resync(self):
        for t in _R3_TABLES:
            t._sync(force=True)

    def add(self, x):
        set.add(self, x)
        self._resync()

    def discard(self, x):
        set.discard(self, x)
        self._resync()

    def remove(self, x):
        set.remove(self, x)
        self._resync()

    def clear(self):
        set.clear(self)
        self._resync()

    def update(self, *xs):
        set.update(self, *xs)
        self._resync()

    def difference_update(self, *xs):
        set.difference_update(self, *xs)
        self._resync()

    def pop(self):
        x = set.pop(self)
        self._resync()
        return x

    def __ior__(self, o):
        self.update(o)
        return self

    def __isub__(self, o):
        self.difference_update(o)
        return self


R3_OFF = _R3Off()
# fr-moe: moe_null_experts ships OFF (the release freeze): measured only on effectively untrained checkpoints
# (Dots1-Dev, tiny_glm4_moe_2.8B: 100% / 24 of 24); the trained pair (Moonlight-16B-A3B) did not complete on the shared
# machine. Discard it from R3_OFF to switch the rule on.
R3_OFF.add("moe_null_experts")


def _r3(spec):
    return spec not in R3_OFF


# --- wall audit (branch wall-audit, dist/wall_audit.md): fixes found by the audit, each behind its own key (WA_OFF = the
# keys switched off):
#   wa_pack_zero_room       soundness: KV-head packing of a position-free target (NoPE / ALiBi / learned-abs) on a
#                           fixed-rotary (partial) RoPE host needs every packed block in the host's zero-frequency dims
WA_KEYS = ("wa_pack_zero_room",)
WA_OFF = set()


def _wa(key):
    return key not in WA_OFF


class _R3Set(set):
    """A type table with round-3 additions (add={spec: entries}) and removals (drop={spec: entries}): the set's own
    content is the live table (base + the additions - the removals of every spec not in R3_OFF), resynced whenever
    R3_OFF changes."""

    def __init__(self, base=(), add=None, drop=None):
        super().__init__()
        # round 3, batch 3a: a base that is itself a round-3 table (a table wrapped twice: FEATURES) is read LIVE, so
        # R3_OFF still switches the inner table's specs off (a frozen copy would keep them on)
        self.__dict__.update(r3_base=frozenset(base), r3_src=base if isinstance(base, _R3Set) else None,
                             r3_add=dict(add or {}), r3_drop=dict(drop or {}), _key=None)
        _R3_TABLES.append(self)
        self._sync(force=True)

    def _sync(self, force=False):
        key = frozenset(R3_OFF)
        if not force and self.__dict__.get("_key") == key:
            return
        src = self.__dict__.get("r3_src")
        out = src._live() if src is not None else set(self.r3_base)
        for s, xs in self.r3_add.items():
            if _r3(s):
                out |= set(xs)
        for s, xs in self.r3_drop.items():
            if _r3(s):
                out -= set(xs)
        set.clear(self)
        set.update(self, out)
        self.__dict__["_key"] = key

    def _live(self):
        self._sync()
        return set(set.__iter__(self))

    def __contains__(self, x):
        self._sync()
        return set.__contains__(self, x)

    def __iter__(self):
        self._sync()
        return iter(list(set.__iter__(self)))

    def __len__(self):
        self._sync()
        return set.__len__(self)

    def __or__(self, other):
        return self._live() | set(other)

    __ror__ = __or__

    def __and__(self, other):
        return self._live() & set(other)

    __rand__ = __and__

    def __sub__(self, other):
        return self._live() - set(other)

    def __rsub__(self, other):
        return set(other) - self._live()

    def __xor__(self, other):
        return self._live() ^ set(other)

    __rxor__ = __xor__

    def isdisjoint(self, other):
        return self._live().isdisjoint(other)

    def issubset(self, other):
        return self._live().issubset(other)

    def issuperset(self, other):
        return self._live().issuperset(other)

    def union(self, *o):
        return self._live().union(*o)

    def intersection(self, *o):
        return self._live().intersection(*o)

    def difference(self, *o):
        return self._live().difference(*o)

    def __eq__(self, other):
        return self._live() == (set(other) if isinstance(other, (set, frozenset)) else other)

    def __hash__(self):
        return id(self)


class _R3Dict(dict):
    """A table dict with round-3 entries (add={spec: {key: value}}, overriding the base) and removals
    (drop={spec: keys}); its own content is the live table, resynced whenever R3_OFF changes."""

    def __init__(self, base=(), add=None, drop=None):
        super().__init__()
        self.__dict__.update(r3_base=dict(base), r3_src=base if isinstance(base, _R3Dict) else None,
                             r3_add=dict(add or {}), r3_drop=dict(drop or {}), _key=None)      # (r3_src: as _R3Set)
        _R3_TABLES.append(self)
        self._sync(force=True)

    def _sync(self, force=False):
        key = frozenset(R3_OFF)
        if not force and self.__dict__.get("_key") == key:
            return
        src = self.__dict__.get("r3_src")
        out = src._live() if src is not None else dict(self.r3_base)
        for s, kv in self.r3_add.items():
            if _r3(s):
                out.update(kv)
        for s, ks in self.r3_drop.items():
            if _r3(s):
                for k in ks:
                    out.pop(k, None)
        dict.clear(self)
        dict.update(self, out)
        self.__dict__["_key"] = key

    def _live(self):
        self._sync()
        return dict(dict.items(self))

    def __getitem__(self, k):
        self._sync()
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._sync()
        return dict.get(self, k, default)

    def __contains__(self, k):
        self._sync()
        return dict.__contains__(self, k)

    def __iter__(self):
        self._sync()
        return iter(list(dict.__iter__(self)))

    def __len__(self):
        self._sync()
        return dict.__len__(self)

    def keys(self):
        return self._live().keys()

    def values(self):
        return self._live().values()

    def items(self):
        return self._live().items()

    def __eq__(self, other):
        return self._live() == other

    def __hash__(self):
        return id(self)

# --- per-slot partial orders: value -> [(parent_value, currency, exact)] -------
# "A -> parent" means A is a special case of parent (parent can emulate A).
# The abstract "*_top" values are the per-slot suprema (used only by --abstract):
# LayerNorm and RMSNorm are incomparable to each OTHER but both <= AffineNorm.
POS_ORDER = {
    "nope": [("partial_rope", "free", True), ("learned_abs", "free", True), ("alibi", "free", True)],
    # alibi -> learned_abs: a per-position table b(j) = -m j on a reserved key channel read by a
    # constant query channel reproduces the ALiBi ramp exactly over the host's bounded context
    # (attn/positional_cross.py cross_matrix: alibi <- learned_abs residual 0.0000; the a(i) row
    # term is softmax-invariant). Currency: the reserved channel (width), bounded T.
    # alibi -> partial_rope: the seed construction needs ONE rotating pair at a small angle, which a
    # partial-RoPE host has on its rotary sub-space (its lowest-frequency pair); same mag limit as rope.
    "alibi": [("rope", "mag", False), ("partial_rope", "mag", False), ("additive_bias", "free", True)],
    # round 2, pass 5: alibi -> learned_abs REMOVED. Under a pre-norm the key-side term m j must reach the logit
    # unscaled, but a position channel read through the norm gives j / R_j (proofs/PROOFS.md "Pre-norm
    # rescue attempts": const_readout_channel, no_prenorm_homogeneous_channel -- refuted as exact; the position x
    # compensator approximation is not run end to end). attn/positional_cross.py's 0.0000 is the norm-free toy.
    "t5_relbias": [("additive_bias", "free", True)],
    "partial_rope": [("rope", "free", True)],   # RoPE on a subspace <= full RoPE
    # rope -> partial_rope: a partial-RoPE host reproduces a full-RoPE head EXACTLY when its rotated
    # dims per head hold the target's rotating pairs (hf/partial_rope_host.py): target pairs onto host
    # pairs (a q / k column permutation: half-split or interleaved), the frequencies in the host's
    # inv_freq buffer (as NoPE <= RoPE sets it), the softmax scale and YaRN mscale folded into W_q. A
    # ROT-GATED edge (currency "rot"): emul_matrix pays it only where rope_fits(target, host), and applies
    # the same gate to a partial target on a partial host (the same construction).
    "rope": [("partial_rope", "rot", True), ("pos_top", "free", True)], "learned_abs": [("pos_top", "free", True)],
    "additive_bias": [("pos_top", "free", True)],
    # a FIXED sinusoid table (XGLM, CTRL): only the abstract top (the universal host's settable learned-abs table
    # holds it) for now; the learned_abs edge for real hosts is left to a later pass (a conservative wall)
    "sinusoidal": [("pos_top", "free", True)],
}
# The rotated-dims gate: a RoPE-family target on a RoPE-family host (full or partial) needs every target
# frequency schedule's rotated dims per head (dims_of rope_sched) to fit in the host's inv_freq tables
# (rope_cap), several schedules sharing a table summing their dims, and its NoPE layers' zero-frequency
# slots too (negative entries): rope_fits. A stock model has one table for all its layers (Gemma-3 / -4,
# Laguna, MiMo, DeepSeek-V4: one per layer type), so a one-table host does not play a target with several
# RoPE tables unless one table holds them all (this replaced the edge-case audit's stopgap multi_rope
# feature). An MLA host's table is its ONE shared rotary key head (rope_cap = qk_rope_head_dim): the gate
# is the MLA <- MLA rotary-key condition; a plain (GQA / MHA) target on an MLA host stays a wall whatever
# rope_fits says -- the latent norm cannot be switched off and one shared rotary key cannot form n_kv > 1
# rotating KV heads (KV_ORDER below).
ROT_HOSTS = frozenset({"rope", "partial_rope"})
ROT_TARGETS = frozenset({"rope", "partial_rope"})
# round 3, batch 2b (rope_cap_zero_tail_table): the remote LFM3 rotary (modeling_lfm3.py:94-111, :127-146; the same
# line in every LFM3 / LFM3-RD / LFM3-Omni copy) builds ONE inv_freq table over the WHOLE head: theta^(-2i/head_dim) for
# i < int(partial * head_dim) // 2, then zeros(head_dim // 2 - rope_angles), cos / sin head_dim wide (:115-124) -- a
# zero-frequency (NoPE) tail, Gemma-4's "proportional" layout. Its config partial_rotary_factor only moves where the
# tail starts (no q / k split, no shape), so the family is fraction-free (Lean partial_rotary_raise_buffer, PROOFS.md
# "free config: raising partial_rotary_factor") and its tables' cap is the whole head. Only the remote code (an auto_map
# to modeling_lfm3*): transformers-main's lfm3 default init ignores the factor (a different defect, not this one).
ZERO_TAIL_ROPE_TYPES = _R3Set(add={"rope_cap_zero_tail_table": {"lfm3", "lfm3_rd", "lfm3_omni"}})
# round 3, batch 2b (rotary_dim_ignored): BailingMoeV2's remote code (modeling_bailing_moe_v2.py:459-460, :204-206)
# builds its table through ROPE_INIT_FUNCTIONS, whose default init reads only partial_rotary_factor (tf4
# modeling_rope_utils.py:124-126): a saved `rotary_dim` is never read (Ling-1T / Ring-1T: rotary_dim 64 of a 128-dim
# head, no factor -> the whole head rotates)
ROTARY_DIM_UNREAD_TYPES = _R3Set(add={"rotary_dim_ignored": {"bailing_moe"}})


def _rotary_dim_unread(d, outer=None):
    """The config's rotary_dim is unread by its code (ROTARY_DIM_UNREAD_TYPES, the BailingMoeV2 remote code)."""
    mt = str(d.get("model_type") or "").lower()
    return mt in ROTARY_DIM_UNREAD_TYPES and "bailing_moe_v2" in _auto_map_str(d, outer)


def _zero_tail_rope(d, outer=None):
    """The remote LFM3 rotary: one whole-head table with a zero-frequency tail (ZERO_TAIL_ROPE_TYPES)."""
    mt = str(d.get("model_type") or "").lower()
    return mt in ZERO_TAIL_ROPE_TYPES and "modeling_lfm3" in _auto_map_str(d, outer)
# MQA <= GQA <= MHA are exact (tile the shared heads) and size-based (kv_cap). MLA is OFF the chain:
# the norm-free toy absorbs its up-projections exactly (attn/mla_rank.py), but real MLA (DeepSeek-V2 /
# V3, Kimi-K2, GLM-4.7-Flash / GLM-5, Mistral-4, MiniCPM3, Youtu, ...) RMS-normalises its compressed KV
# latent per token (kv_a_layernorm; the query latent too when q_lora_rank: q_a_layernorm) before the
# up-projection, so its keys and values are NOT linear in the normed residual:
#   MLA target <- plain host: the best a plain host computes is the norm folded to one constant scale per
#     layer. MEASURED on DeepSeek-V2-Lite (attn/mla_norm_agreement.py; attn/data/mla_norm_agreement.jsonl):
#     the latent's rms varies 0.4 - 11x across tokens (per-layer cv 0.17 - 1.6) and the folded model fails
#     the act/act_swap_agreement.py standard by far (top-1 81%, 4% identical: MLA_FOLD below). Not payable.
#   plain target <- MLA host: the host's latent norm cannot be switched off (only its gain is free), and its
#     rotary part is ONE key head of qk_rope_head_dim dims shared by every query head: a RoPE target with
#     more than one rotating KV head has keys the host cannot form at all. A wall.
#   MLA target <- MLA host: exact when the host's latents are at least as wide (a zero-padded latent's rms
#     is a constant multiple of the target's: the ratio folds into the norm gain), the query latent norm is
#     present in both or neither, and the shared rotary key is at least as wide (emul_matrix: kv_latent,
#     q_latent, mla_rope).
KV_ORDER = {"MQA": [("GQA", "kv", True)], "GQA": [("MHA", "kv", True)]}
MLA_FOLD_PAYABLE = False
# the sequential host's carry of a parallel block's input under a pre-norm (block/prenorm_carry.py: ~1e-2):
# not payable, like the CLA carry (mode="mixgap" keeps it)
PAR_CARRY_PAYABLE = False
# DeepSeek-V2-Lite, all 27 kv_a_layernorms folded, 24 prompts, fp32 (CPU): top-1 81.0%, 4.2% of 32-token
# continuations identical, first divergence at token 4.5 on average; rms(c) per-layer cv mean 0.58, max 1.58
MLA_FOLD = dict(model="deepseek-ai/DeepSeek-V2-Lite", variant="kv", top1=0.810, ex32=0.042, cv_mean=0.58)
FFN_ORDER = {"vanilla": [("glu", "free", True)], "glu": [("moe", "cap", True)],
             "moe": [("moe_shared", "free", True)]}
# round 3 (mamba_ffn_none): a block with NO FFN (the pure Mamba stacks: transformers MambaBlock / FalconMambaBlock /
# Mamba2Block are norm + mixer + residual, modeling_mamba.py:478-503, modeling_falcon_mamba.py:543, modeling_mamba2.py
# :683) is the FFN value "none", below vanilla: any FFN host plays it by zeroing its MLP's output projection (down_proj
# = 0, every expert's and the shared one's for a MoE: the sublayer adds 0, Lean zeroed_sublayer), exact; a "none" host
# plays only "none". Inert with the spec off (no point carries the value: classify gives them vanilla / glu then)
FFN_ORDER["none"] = [("vanilla", "zeroed_sublayer", True)]
# RMSNorm emulates LayerNorm EXACTLY by folding the mean-centering into the
# adjacent linear layer (column-centered weight constraint) -- proven exact for
# train+inference (arXiv:2605.14521, FlashNorm). So norm is DIRECTIONAL, not
# incomparable: LayerNorm -> RMSNorm (fold). The reverse needs an extra dim to
# carry the mean, so RMSNorm is the more general (top) norm.
# The reverse is a WIDTH edge: a LayerNorm host reproduces an RMSNorm target EXACTLY on the mirrored
# residual [h; -h] (mean 0, so the centring is a no-op and std = rms; every writer writes [w; -w]),
# at 2x hidden (tests/test_composed_edges.py::test_norm_rmsnorm_is_layernorm_at_double_width).
# emul_matrix charges the 2x hidden (NORM_WIDTH_FACTOR).
NORM_ORDER = {"LayerNorm": [("RMSNorm", "fold", True)],
              "RMSNorm": [("AffineNorm", "free", True), ("LayerNorm", "width", True)]}
NORM_WIDTH_FACTOR = 2
# --- activations: the measured single-op minimax floor matrix, thresholded -----------------
# ACT_FLOORS[target][host] = floor_minimax(host, target) from act/activation_cross.py (the true
# single-op minimax: best a*X(c*x + d) + e*x + f against Y on [-6, 6], max |error|), recorded
# 2026-11-11. tests/test_activation_cross.py re-measures the cells the threshold depends on.
# THRESHOLD RULE: host X pays for target Y iff its floor is <= ACT_FLOOR_MAX = 0.0005, CALIBRATED
# on real models (act/act_swap_agreement.py; data act/data/act_swap_agreement.jsonl; table, thresholds
# and figure: python -m act.act_swap_report). Every FFN activation of a real checkpoint is replaced by
# the host's best fit a*X(c*x + d) + e*x + f and scored on the ORIGINAL model's own greedy 64-token
# continuations. Payable = de facto high-agreement generation, defined as top-1 agreement >= 99% AND
# >= 90% of 32-token greedy continuations token-identical. Measured: every 0.000 cell (the exact
# folds and the beta -> inf limits) passes at 100% / 100% on every model; every cell at floor >= 0.009
# fails on every model (the swish cluster GeLU / SiLU / mish 0.009 - 0.018: top-1 92 - 99% and 0 - 79%
# identical on most models, 2 - 26% on Qwen2.5-1.5B; 0.07 - 0.14, e.g. SiLU <- ReLU 0.137: top-1
# 14 - 74%, 0% identical). Note the trained Apertus xIELU (per-layer alphas) is much farther from
# every host than the default-init xIELU row below (floors 1.4 - 34, top-1 <= 16%). Scaled fits
# Y + t (fit - Y) bracket the boundary: floor 0.0004 passes, 0.001 fails, so 0.0005. The relation is
# the same for any threshold in [0, 0.009) -- and every looser definition tried brackets at 0.006 -
# 0.010 too: "no worse than the original in bf16", "top-1 >= 98% and >= 50% identical", even a
# teacher-forced "top-1 >= 95%" alone (SiLU <- mish 0.010 gives 1.6% on Qwen2.5-1.5B, whose
# pre-activations leave [-6, 6]). The previous 0.15 band was not payable.
# A 0.000 cell is either an exact fold (SiLU <-> swigluoai by the 1.702 scale) or a beta -> inf LIMIT
# (ReLU from a swish; relu2 from xIELU as c -> inf), labelled by ACT_EXACT / ACT_LIMIT below.
# Floors compound under composition, so the act relation is the one-hop table, NOT its transitive
# closure (REACH["act"] below).
ACT_FLOORS = {
    "ReLU":      {"relu2": 0.374, "SiLU": 0.000, "GeLU": 0.000, "mish": 0.000, "swigluoai": 0.000,
                  "situ": 0.161, "xielu": 0.374, "GeLU_tanh": 0.000},
    "relu2":     {"ReLU": 1.807, "SiLU": 0.207, "GeLU": 0.296, "mish": 0.242, "swigluoai": 0.207,
                  "situ": 0.216, "xielu": 0.000, "GeLU_tanh": 0.295},
    "SiLU":      {"ReLU": 0.137, "relu2": 0.316, "GeLU": 0.014, "mish": 0.010, "swigluoai": 0.000,
                  "situ": 0.071, "xielu": 0.316, "GeLU_tanh": 0.014},
    "GeLU":      {"ReLU": 0.084, "relu2": 0.373, "SiLU": 0.012, "mish": 0.018, "swigluoai": 0.012,
                  "situ": 0.131, "xielu": 0.373, "GeLU_tanh": 0.0003},
    "mish":      {"ReLU": 0.108, "relu2": 0.360, "SiLU": 0.009, "GeLU": 0.020, "swigluoai": 0.009,
                  "situ": 0.108, "xielu": 0.360, "GeLU_tanh": 0.020},
    "swigluoai": {"ReLU": 0.081, "relu2": 0.367, "SiLU": 0.000, "GeLU": 0.012, "mish": 0.010,
                  "situ": 0.125, "xielu": 0.367, "GeLU_tanh": 0.012},
    "situ":      {"ReLU": 0.317, "relu2": 0.323, "SiLU": 0.286, "GeLU": 0.188, "mish": 0.288,
                  "swigluoai": 0.214, "xielu": 0.484, "GeLU_tanh": 0.188},
    "xielu":     {"ReLU": 1.641, "relu2": 0.136, "SiLU": 0.135, "GeLU": 0.174, "mish": 0.162,
                  "swigluoai": 0.135, "situ": 0.141, "GeLU_tanh": 0.173},
    # the tanh-approximate GeLU (gelu_new / gelu_pytorch_tanh / ...), split from the erf "GeLU" (round 2): floors
    # floor_minimax on [-6, 6], 2026-11-13. The GeLU <-> GeLU_tanh cells (0.0003 both ways: 2.69e-4 fitted, 4.73e-4
    # raw) are measured payable both ways in round 2, pass 5 (graph/gelu_variants.py, act/data/gelu_swap_agreement.jsonl:
    # gpt2, gpt2-medium, Pythia-160m / 410m / 1.4b, Gemma-2-2b, raw swap, eager attention): approximate floors.
    "GeLU_tanh": {"ReLU": 0.084, "relu2": 0.373, "SiLU": 0.012, "mish": 0.018, "swigluoai": 0.012,
                  "situ": 0.131, "xielu": 0.373, "GeLU": 0.0003},
}
ACT_FLOOR_MAX = 0.0005
ACT_EXACT = set()     # (target, host). swigluoai <- SiLU is a 0.000 cell but NOT exact: g sigmoid(1.702 g) =
#   SiLU(1.702 g) / 1.702 folds, yet the (up + 1) needs an up-projection bias (FEATURES bias_out) and gpt-oss's
#   +-7 clamps hold only inside the clamp regime (Lean swigluoai_sub_silu): a regime, in ACT_LIMIT
ACT_LIMIT = {("ReLU", h) for h in ("SiLU", "GeLU", "GeLU_tanh", "mish", "swigluoai")} | {
    ("relu2", "xielu"),                        # xIELU(c x) / (alpha_p c^2) -> relu(x)^2 as c -> inf
    ("SiLU", "swigluoai"), ("swigluoai", "SiLU")}   # the 1.702 scale folds, but the +-7 clamp needs the input
#                                                scaled inside it (magnitude)
# poly_norm (Motif-3) has no measured cell and no construction: a wall.


def _act_order(floor_max=None):
    fm = ACT_FLOOR_MAX if floor_max is None else floor_max
    order = {}
    for tgt, row in ACT_FLOORS.items():
        for host, fl in sorted(row.items(), key=lambda kv: kv[1]):
            if fl <= fm:
                cur, ex = (("fold", True) if (tgt, host) in ACT_EXACT else
                           ("mag", False) if (tgt, host) in ACT_LIMIT else ("floor", False))
                order.setdefault(tgt, []).append((host, cur, ex))
    order.setdefault("SiLU", []).append(("act_top", "free", True))      # the abstract top (--abstract)
    return order


ACT_ORDER = _act_order()

# --- "small config" slots (block structure + attention details) --------------
# Block: parallel is a RESTRICTION of sequential (drop the attn->MLP dependency so
# they run concurrently). SEQUENTIAL is more general -- it emulates parallel with
# extra WIDTH at the SAME depth (carry pre-attn x in reserved channels, MLP reads
# them). The reverse (parallel emulates sequential) costs 2x DEPTH. Both simulate
# each other, so block is size-based in emulates(); this order is the type view.
# qk-norm and softcap are APPROXIMATELY foldable, not must-match:
#   qk-norm  folds to a fixed scale when ||q|| is stable  (attn/qk_norm.py)
#   softcap  ~= identity when logits << cap                (attn/softcap.py)
# so a plain (no) model approximately emulates a capped / normed one (approx; REGIME_PAYABLE prices it).
BLOCK_ORDER = {"parallel": [("sequential", "width", True)]}   # sequential is more general
WINDOW_ORDER = {"local": [("full", "mask", True)]}            # full emulates local by masking
# qk-norm / softcap are REGIME edges, and the residual is the same expression in both directions:
#   softcap: c tanh(x / c) vs x -- a plain host for a softcap target and a softcap host for a plain
#     target (same logits) differ by exactly |c tanh(x/c) - x| (attn/softcap.py), small when |x| << c;
#   qk-norm: q / rms(q) vs q / c_q -- a plain host folds the target's norm into a constant 1/c_q, a
#     qk-norm host sets its gain to c_q; both residuals are |q/rms(q) - q/c_q| up to the scale c_q,
#     small when ||q|| is stable across tokens (attn/qk_norm.py; both directions in
#     tests/test_hf_modules.py::test_qknorm_softcap_regime_is_symmetric).
# qk-norm VARIANTS (round 2, pass 2; literature review, "qk-norm sub-variants", code-
# checked in transformers 5 and the remote modeling files). They are different functions, not one "yes":
#   head            per-head RMSNorm with one head_dim gain, before RoPE (Qwen3, Gemma-3 / 4, LFM2, Apertus,
#                   EXAONE-4, GLM-4.5, Ling-2, Hy3, Step-3.5, OpenELM, ...)
#   head_distinct   per-head RMSNorm with a gain PER HEAD (num_heads, head_dim): PLaMo-2
#   head_free       weight-free per-head RMSNorm (Llama-4 L2Norm and nanochat after RoPE, ARLoop before it: the rms is
#                   rotation-invariant, so the order does not matter without a gain)
#   head_post       per-head RMSNorm with a gain AFTER RoPE (Hunyuan-v1 dense / MoE, dhara): w * R_m q is not R_m (w q)
#                   unless w is constant on each rotated pair, so no pre-RoPE gain reproduces it
#   proj            one RMSNorm over the WHOLE q / k projection (num_heads x head_dim: OLMo-2 / 3, OLMoE, FlexOlmo,
#                   OLMo-hybrid, MiniMax-M2, Bolmo, Molmo-O): the rms is over every head jointly
#   proj_ln         one LayerNorm over the whole projection (OLMo-1 remote attention_layer_norm, Molmo LayerNorm)
#   head_ln         per-head LayerNorm with per-head-distinct gains, no bias (Cohere use_qk_norm, StableLM-2-12B)
#   head_ln_bias    per-head LayerNorm with per-head-distinct gains and biases (Chameleon)
#   ln_shared_bias  per-head LayerNorm with one shared gain and bias (Persimmon)
#   dsv4            DeepSeek-V4: a weight-free per-head q norm and a weighted norm of the one K = V head
#   qkv_input       CENO qkv_layernorm: a second RMSNorm on the attention INPUT (a per-token renormalisation)
#   unverified:<t>  a q / k norm in remote code not available to check: its own value (walled from the others)
# A target with a variant needs a host with the same one, except where a host's variant contains the target's by
# a weight setting (exact): a gained per-head norm with gain 1 is the weight-free one (head_free <- head,
# head_post, head_distinct); per-head-distinct gains all equal are a shared gain (head <- head_distinct); a
# zero bias / equal per-head weights (head_ln <- head_ln_bias, ln_shared_bias <- head_ln_bias). Between a
# variant and no qk-norm the edge is the regime fold (approx, both ways), measured NOT payable (REGIME_PAYABLE);
# switching a host's qk-norm off is the eps lever, decided in a later pass.
QKNORM_VARIANTS = ("head", "head_distinct", "head_free", "head_post", "proj", "proj_ln", "head_ln", "head_ln_bias",
                   "ln_shared_bias", "dsv4", "qkv_input")
QKNORM_NEST = {"head_free": ("head", "head_post", "head_distinct"), "head": ("head_distinct",),
               "head_ln": ("head_ln_bias",), "ln_shared_bias": ("head_ln_bias",)}
# round 3, batch 2a: five more variants, each only ever a value of the configs its spec classifies (so the orders below
# are inert with that spec in R3_OFF: no point carries the value) --
#   head_qfree      weight-free per-head q norm, weighted per-head k norm, before RoPE (Domyn-Edge: q_norm
#                   RMSNorm(elementwise_affine=False), k_norm weighted, modeling_domynedge.py:223-224, :238-239;
#                   qknorm_by_code_unlisted)
#   head_free_gain  weight-free per-head q / k norm times a CONFIG constant c on the logit (nanochat_gpt
#                   qk_sharpen_scale^2, gaggle qk_scale^2, sn38-nanochrono attention_multiplier^2, Grug-MoE qk_mult,
#                   Muse-Glimmer qk_scale_factor: qknorm_fixed_gain / qknorm_by_code_unlisted)
#   ngpt            Dory's weight-free L2 norms with a per-channel q gain (sqk sqk_scale)^2 AFTER RoPE (modeling_dory.py
#                   :350-351, :383-384; qknorm_by_code_unlisted)
#   dsv4_kv         DeepSeek-V4.1: only the weighted norm of the one K = V head, V4's weight-free q norm gone
#                   (inference/model.py:771-773; dsv41_q_head_norm_absent)
#   mla_krope       dots3-note: a weighted RMSNorm on the shared rotary key k_rot of MLA before RoPE, nothing on q
#                   (modular_dots3_note.py:489, :516-523; mla_k_rope_norm)
# Nestings (exact, a weight / config setting of the host): head_free <- head_qfree (its k gain 1: Lean qkn_gain_one;
# the weight-free q commutes with RoPE, weightFree_comm), head_free <- head_free_gain (its gain key set to 1: a free
# config scale; the rest is the weight-free norm), head_qfree <- head / head_distinct (q gain 1, k gain the target's:
# qkn_gain_one, qkn_equal_gains). WALLS until their theorems land: head_free_gain <- head / head_post / head_distinct
# (w = c 1 on q, needs theorem qkn_const_gain) and head_free <- ngpt / head_free_gain <- ngpt (an L2 norm with a
# per-channel gain after RoPE set to the RMS constant sqrt(d): qkn_const_gain again, and its eps). ngpt, dsv4_kv
# (dsv4 <-> dsv4_kv: needs theorem q_headnorm_not_plain for the wall's record) and mla_krope nest only in themselves.
QKNORM_VARIANTS = QKNORM_VARIANTS + ("head_qfree", "head_free_gain", "ngpt", "dsv4_kv", "mla_krope")
# round 3, batch 2d (hybrid_mla_layers_unread): q_head = Sabi's weighted per-head q norm with no k norm (modeling_sabi_mla
# .py:93-94, :121-127: k_norm deleted with k_proj / v_proj); nests only in itself (the mla_qmix feature walls it anyway)
QKNORM_VARIANTS = QKNORM_VARIANTS + ("q_head",)
QKNORM_NEST = {**QKNORM_NEST, "head_free": QKNORM_NEST["head_free"] + ("head_qfree", "head_free_gain"),
               "head_qfree": ("head", "head_distinct")}
QKNORM_ORDER = {v: [(h, "fold", True) for h in QKNORM_NEST.get(v, ())] + [("no", "approx", False)]
                for v in QKNORM_VARIANTS}
QKNORM_ORDER["no"] = [(v, "approx", False) for v in QKNORM_VARIANTS]
# softcap, split by WHERE the cap acts (round 2, pass 2): "attn" = a cap c tanh(z / c) on the attention scores
# (Gemma-2 attn_logit_softcapping; with or without a final cap: a capped host sets its final cap large or a final-
# capped target drops it, both payable below, so the two merge), "final" = only the output logits are capped
# (Gemma-3n / 4, nanochat, RecurrentGemma, rnj-1, ...), "no" = neither. Edges: every direction is a regime edge
# (approx) for the type lattice; the headline pays only the measured ones (REGIME_PAYABLE).
SOFTCAP_ORDER = {"attn": [("no", "approx", False), ("final", "approx", False)],
                 "final": [("no", "argmax", False), ("attn", "cap-large", False)],
                 "no": [("final", "cap-large", False), ("attn", "cap-large", False)]}
# REGIME_PAYABLE[(slot, target, host)]: is the regime edge payable under the headline? MEASURED on real
# checkpoints, both directions (attn/regime_agreement.py; data attn/data/regime_agreement.jsonl; table:
# python -m attn.regime_agreement --report), with the activation-swap standard: the construction swapped
# into the real model, scored on the ORIGINAL model's own greedy 64-token continuations of 24 prompts,
# fp32, TF32 off; payable = top-1 >= 99% AND >= 90% of 32-token greedy continuations identical.
#   softcap, capped target on a plain host (drop the caps): NOT payable for an ATTENTION cap. The final-logit
#     cap is monotone, so dropping it leaves every argmax alone (100% / 100% on Gemma-2-2B / -2b-it / -2b-jpn-it
#     / -9B): final <- no is payable (exact for greedy). The ATTENTION cap is what bites. Gemma-2-2B passes
#     narrowly (top-1 99.7%, 92% identical; -it and -jpn-it 96%) because its scores stay under the cap (max |z|
#     23 - 27 vs c = 50), but Gemma-2-9B's reach 66 (layer 24) and it fails: 99.2%, 75% identical. It passes
#     only the looser definitions (top-1 >= 98% and >= 50% identical; no worse than the same model in bf16,
#     98.6% / 62%). The snapshot's attention-capped targets are Gemma-2 2B / 9B / 27B, so attn <- no and
#     attn <- final (a final-capped host has no attention cap) are refused.
#   softcap, plain target on a capped host: payable, with the host's cap c set large (c is a config
#     scalar, like the RoPE tables the rot gate sets). Six plain checkpoints pass from c = 50 - 5,000,
#     always 1.1 - 4.6x the target's own max |z| (Llama-3.2-1B 46 -> c >= 50; SmolLM2-135M 53 -> 100;
#     OLMo-2-1B 22 -> 100; Gemma-3-270m 44 -> 100; Qwen3-0.6B 48 -> 200; Qwen2.5-0.5B, whose
#     attention-sink scores reach 1,728, -> 5,000), and |c tanh(z/c) - z| <= |z|^3 / (3 c^2) closes
#     the gap from there (c >= 5 max |z| is the rule). Precision does not limit c: in fp32 the residual
#     reaches the fp32 floor (KL ~1e-11 at c >= 5,000), and in bf16 a large cap gives exactly the bf16
#     control's agreement. At Gemma-2's native c = 50 it is not payable (SmolLM2 83%, OLMo-2 88%, Qwen3
#     46%, Qwen2.5 0% identical). A final-logit cap of 15 or 30 is 100% / 100% on all six (c = 5 saturates
#     fp32 into argmax ties: 88 - 100%). So no <- attn, no <- final, final <- attn are payable.
#   qk-norm, either way: NOT payable. A fixed scale cannot stand in for a per-token 1/rms(q), 1/rms(k):
#     the rms varies across tokens (coefficient of variation 0.11 - 0.44 per head, up to 1.6; the
#     attention-sink token above all). qk-normed target on a plain host, the norm folded to its
#     least-squares constant per head: Gemma-3-270m 49% top-1, 0% identical (fit on the evaluation text
#     itself; 49% on held-out text), Qwen3-0.6B 15%, Qwen3-1.7B 12%, OLMo-2-1B 27%, all 0% identical
#     (block/fold_agreement.py, the edge-case audit). Plain target on a qk-norm host with the
#     least-squares gain per head (per-head or whole-projection norm): SmolLM2-135M 41%, Qwen2.5-0.5B
#     42%, Llama-3.2-1B 15%, Gemma-2-2B 25%, all 0% identical. (The variants' exact nestings, QKNORM_NEST,
#     are not regime edges: they are paid.)
# Every mode but "exact" (all regime edges refused) and "capacity" (slots ignored) pays exactly these.
REGIME_PAYABLE = {("softcap", "attn", "no"): False, ("softcap", "attn", "final"): False,
                  ("softcap", "final", "no"): True, ("softcap", "no", "final"): True,
                  ("softcap", "no", "attn"): True, ("softcap", "final", "attn"): True,
                  **{("qknorm", v, "no"): False for v in QKNORM_VARIANTS},
                  **{("qknorm", "no", v): False for v in QKNORM_VARIANTS}}
REGIME_PAYABLE[("sink", "yes", "no")] = False     # drop the sinks: measured not payable (SINK_ADD_PAYABLE)
REGIME_PAYABLE[("sink", "no", "yes")] = True      # the b -> -inf limit (round 2, pass 5)
REGIME_SLOTS = ("qknorm", "softcap", "sink")

# --- more axes (sequence mixer, sinks, cross-layer KV, MTP, generation) --------
# softmax attention is the general mixer TYPE: an SSM / linear / hybrid recurrence reduces into it
# (MIXER_ORDER drives the per-slot plans of graph/reduce.py / construct.py and the type lattice) --
# but the near-gap is NOT PAYABLE under the headline (MIXER_GAP_PAYABLE). MEASURED on real models with
# the activation-swap standard (attn/mixer_gap_agreement.py; data attn/data/mixer_gap_agreement.jsonl;
# table / figure: python -m attn.mixer_gap_report): every recurrent mixer of a real checkpoint replaced
# by an explicit softmax construction, scored on the ORIGINAL model's own greedy 64-token
# continuations; payable = top-1 >= 99% AND >= 90% of 32-token continuations identical. Complete
# constructions DO reach it (100% / 100%) -- the softmax normaliser undone by a sink key, decay as a
# logit, small temperature for q.k, hard-selection shifts, GLU products -- but only at a price the
# edge never charged (MIXER_GAP below):
#   * a data-dependent decay (Mamba-1/2, GLA, GDN's g) needs a PRIOR layer writing the prefix sum: a
#     softmax logit q_t.k_s sees tokens t and s only, the decay prod_{s<r<=t} a_r every token between
#     (one layer with a fixed ALiBi decay: mamba-130m 85% top-1, 4% identical; granite 85%, 12%);
#     so do token shifts and pre-conv nonlinearities. Carrying that state across sublayers under a
#     stock single-rms pre-norm is impossible at any width (block/prenorm_carry.py): a norm change;
#   * heads: Mamba-1 one per (channel, state) pair (24,576 for mamba-130m; 16 - 4,096 shared decay
#     rates: top-1 57 - 95%, 0 - 25% identical), RWKV-4 one per channel (768), a one-layer short conv 12,288;
#     the zoo's widest softmax host has 128 query heads. Mamba-1's softplus also needs 256 SiLU units
#     per channel (its |A| reaches ~1e6, so dt errors of 1e-6 move the decay; 64 units: 98.4% / 79%);
#   * the delta rule (gated DeltaNet: Qwen3.5 / Qwen3-Next / Kimi / RWKV-7) has non-commuting
#     transitions; its softmax form is a Jacobi iteration exact only after T - 1 hops -- host depth
#     proportional to CONTEXT (Qwen3.5-0.8B: p = 0..32 hops 3 - 92% top-1, 0 - 54% identical; p = 64
#     98.9% / 100%; p = 96 >= T exact): no fixed host is enough for every context;
#   * a distilled softmax layer at the target's width (the achievability probe,
#     attn/mixer_gap_distill.py) is far off: mamba-130m 2 - 16% top-1 (12 / 48 heads, 600 / 3,000 steps).
# So a softmax host never pays for an SSM / linear / hybrid target in the headline; mode="mixgap" is
# the old relation that granted the near-gap free (frontier 85 instead of 101; 106 since the CLA carry
# is refused too, CLA_CARRY_PAYABLE below).
MIXER_ORDER = {"ssm": [("softmax", "near-gap", False)], "linear": [("softmax", "near-gap", False)],
               "hybrid": [("softmax", "near-gap", False)]}
# branch fr-mixer, rule pure_on_hybrid (dist/fr_mixer.md; Lean FrMixer.padded_run / assign_run): a PURE recurrent stack
# (ssm / linear: every layer a recurrent mixer) runs on any host with recurrent layers -- a hybrid one with its attention
# layers zeroed (zeroed_sublayer: o_proj = 0, and the layer's FFN down_proj = 0, so the layer is the identity), target
# layer i on the host's i-th recurrent layer. Every target layer has one kind, so the host's recurrent layers taken in
# order ARE an order-preserving embedding (the Hall count kind_match checks is then exact, not just a count). The
# family is not the mixer label's: the rf_ features (feats_ok, _rec_ok) check it and the R layer kind counts the depth
# (the pass-4 reason "the mixer slot does not carry the family" is obsolete since the rf_ features). A hybrid target
# keeps its hybrid-only host (its attention layers need attention layers). Only ssm -> hybrid is added: linear ->
# hybrid and the ssm <-> linear label change are the same construction but cross no zoo pair and have no stock
# transformers linear class for a catalog instance, so they stay out (conservative).
FRM_PURE_ON_HYBRID = True
if FRM_PURE_ON_HYBRID:
    MIXER_ORDER["ssm"] = MIXER_ORDER["ssm"] + [("hybrid", "zero the attention layers", True)]
MIXER_GAP_PAYABLE = False
# Per family: the cheapest COMPLETE construction that meets the standard on the real checkpoint, and
# its price per target mixer layer (heads = softmax heads, depth = host layers, groupnorm = needs a
# non-stock pre-norm to carry state between sublayers). verdict "priced": payable at that price, which
# no zoo host pays (> 128 heads or a group norm); "wall": no fixed price (depth ~ context).
# tests/test_mixer_gap.py checks every entry against the recorded rows.
MIXER_GAP = {
    "mamba1":    dict(model="state-spaces/mamba-130m-hf", best="full-construction-sp256", heads=26116,
                      depth=4, groupnorm=True, verdict="priced",
                      construction="sink-normalised exp-decay head per (channel, state); prefix-sum, "
                                   "shift heads; softplus by 256 SiLU units"),
    "rwkv4":     dict(model="RWKV/rwkv-4-169m-pile", best="full-construction", heads=769, depth=2,
                      groupnorm=True, verdict="priced",
                      construction="ALiBi head per channel + DFT-RoPE diagonal bonus; token-shift head; "
                                   "sigmoid by 64 SiLU units"),
    "mamba2":    dict(model="ibm-granite/granite-4.0-h-350m-base", best="full-construction-o2", heads=101,
                      depth=3, groupnorm=True, verdict="priced",
                      construction="2 small-temperature heads per SSD head (q = C, k = B) with sinks; "
                                   "prefix-sum, shift heads; softplus by 64 SiLU units"),
    "shortconv": dict(model="LiquidAI/LFM2-350M", best="onelayer-o2-k0.1", alt="conv16", heads=12288,
                      depth=1, groupnorm=False, verdict="priced",
                      construction="ONE stock layer: per (channel, offset) 4 stencil heads (B, C in the "
                                   "logits); or 3 shift heads x 3 layers + group norm (alt)"),
    "gdn":       dict(model="Qwen/Qwen3.5-0.8B-Base", best="neumann-p96-o4-k0.3", heads=64, depth="T",
                      groupnorm=True, verdict="wall",
                      construction="Jacobi hops of decayed linear attention (4 small-temperature heads per "
                                   "head per hop); exact only at T - 1 hops"),
}
# sink == standard softmax + a zero-value null key at a learned per-head logit b_h (attn/attention_sink.py). A
# sink host plays a plain target exactly (b -> -inf). The reverse is NOT a weight setting: the null key's logit
# must be b_h for EVERY query, and q . k_null is query-dependent unless q has a constant component -- a q bias
# and a rotation-free pair, which a bias-free RoPE host lacks (the toy sets the score directly). So a plain host
# for a learned-sink target is a regime edge (drop the sinks), payable only if measured (SINK_ADD_PAYABLE).
SINK_ORDER = {"yes": [("no", "drop-sinks", False)], "no": [("yes", "mag", False)]}
# round 3, batch 2a (sink_key_query_dependent): sink "key" -- IQuest-Q1's learned sink is a zero-value KEY s per KV head
# (vllm-iquest-q1 model.py:354-365, attention.py:66-89: logit (R_p q) . s, the q normed and rotated, s not), not a
# per-head scalar logit. Isolated both ways (no edge from or to "no" / "yes"): the start-token construction for a key
# target (k0 = R_1 s, v0 = 0 at a host-only position 0) needs theorem sink_key_start_token, and a key host cannot
# remove its sink (q . s -> -inf for every q needs a constant query channel the q-normed host lacks: needs theorem
# sink_key_host_ne_plain for the wall's record). The value only exists with the spec on.
SINK_ORDER["key"] = []
# round 2, pass 5: a sink host plays a plain target only as b -> -inf, a LIMIT (Lean sink_host_factor: the null key's
# weight is e^b / (e^b + sum), never exactly 0), not the exact fold it was labelled; payable (REGIME_PAYABLE), out of
# the exact slice. A sink target on a sink-free host is the start-token rule (emul_matrix: a q bias, a spare slot).
# MEASURED: gpt-oss-tinystories with its learned sinks dropped: top-1 97.5%, 33% of continuations identical
# (block/fold_agreement.py) -- not payable.
SINK_ADD_PAYABLE = False
# CLA: a standard host carries the shared K,V source in reserved channels (attn/cross_layer_kv.py).
# Exact only in the norm-free toy; under a real pre-norm the shared rms rescales the carried copy
# (block/prenorm_carry.py, ~6e-2 on stock Llama layers) -> a WIDTH edge with a FLOOR, not exact.
CLA_ORDER = {"yes": [("no", "width", False)]}
# Not payable under the headline: the ~6e-2 pre-norm floor is two orders above ACT_FLOOR_MAX, the same
# carry problem that makes the softmax <- SSM near-gap unpayable (MIXER_GAP_PAYABLE). mode="mixgap" keeps it.
CLA_CARRY_PAYABLE = False
MTP_ORDER = {"yes": [("no", "free", True)], "no": [("yes", "free", True)]}    # aux predict heads are irrelevant to the MAIN next-token logits -> free both ways
# Auxiliary slots that are free BOTH ways (drop / append the MTP heads; the sink is not: SINK_ORDER):
# they don't touch the main-task forward pass, so they must NOT differentiate classes -- else two
# otherwise-identical models split into a spurious mutual pair. Excluded from the dedup signature.
_AUX_SLOTS = frozenset({"mtp"})
# AR <= diffusion: a bidirectional (diffusion) host given a causal mask and a fixed left-to-right
# unmasking order IS autoregressive attention (attn/ar_diffusion.py, max|d| = 0; the universal host
# toggles exactly this mask). One-way: a causal host cannot see the future in one pass, so a
# diffusion target still needs a diffusion host.
# Round 2, pass 4: BLOCK-causal targets (SDAR: bidirectional inside a block, causal across blocks). SDAR's code
# (modeling_sdar_moe.py) runs non-causal attention under a CALLER-supplied mask (is_causal=False, attn_mask=attention_
# mask; the block mask is built by the generation code from block_size, outside the model). So the same weights run
# any mask: a causal mask is AR (block size 1), an all-ones mask is the bidirectional forward. Hence
#   ar <= block            (mask: block size 1 = causal)
#   block <= diffusion     (mask: a full-attention host given the block mask, the same currency as ar <= diffusion)
#   diffusion <= block     (mask: an all-ones mask; equivalently a block as long as the context)
# -- block and diffusion are one class under the mask currency; the AR host still plays neither (its mask is
# causal in code). The universal host builds only a causal or a full mask (elephantine/universal.py UAttention): a
# block / caller-supplied mask is an OPEN HOST BRANCH (minimal_emulator FAMILY_STATUS "block-causal"), not a relation
# issue.
GEN_ORDER = {"ar": [("diffusion", "mask", True), ("block", "mask", True)],
             "block": [("diffusion", "mask", True)],
             "diffusion": [("block", "mask", True)]}
# round 3 (prefix_lm_mask): a prefix LM (bidirectional over the prompt, causal after it: DIVE-Doc) under the same mask
# currency: a prefix host takes the caller's 4D mask (modeling_divedoc.py:261-263), so it plays causal / block /
# bidirectional targets; a bidirectional or block host given the prefix mask plays it (Lean pattern_sub_full,
# block_sub_diffusion, causal_stack); a causal host cannot (causal_ne_bidirectional). Inert unless a point has gen
# "prefix" (only classify's prefix_lm_mask rule gives it): the reach sets of ar / block / diffusion are unchanged.
GEN_ORDER["ar"].append(("prefix", "mask", True))
GEN_ORDER["prefix"] = [("diffusion", "mask", True), ("block", "mask", True)]
GEN_ORDER["block"].append(("prefix", "mask", True))
GEN_ORDER["diffusion"].append(("prefix", "mask", True))

# round 3 (nonlm_heads; dist/r3_nonlm.md on branch host-nonlm, a design decision on readouts): what a model OUTPUTS
# (`head`: lm | score | embedding | value | flow) and what it READS (`input`: tokens | patches) are two categorical
# slots, one value per model (an LM is lm / tokens). THE READOUT RULE (binding): the emulator's output may be read as a
# FIXED subset of coordinates at FIXED positions -- the last / first token, the per-token rows -- the way decoding reads
# logits; no host computation, no per-input choice. Mean pooling and L2 normalisation are NOT such reads (a function of
# every position / of the whole vector). So
#   score     <- lm   READOUT (exact): an UNTIED host's lm_head row := the score head (the rerankers' Linear(hidden, 1):
#                     modeling_minicpm_reranker.py:1134-1142, gemma_model.py:485-511; read at the final norm, the
#                     default call's one exit cutoff_layers = [L]: :1256-1257 / :595-596), the read is that row at the
#                     token the caller scores. A head bias needs the host's lm_head bias; a tied host's lm_head IS its
#                     table (FEATURES tied_emb, host side a wall), so it cannot set an independent head
#   embedding <- lm   READOUT (exact) for a last- / first-token embedding without L2 normalisation (readout_sig "last" /
#                     "first"): lm_head rows := the Dense (or the identity on d rows), the same untied / bias conditions.
#                     Every embedding model of the snapshot pools by the mean or normalises (EMBED_READOUTS), so this
#                     credits none of them: they are walls against LMs (_r3_masks_nonlm), carried only by the join
#   value, flow       walls both ways (a ReLU MLP after the final norm: an LM host has no nonlinearity there; the flow
#                     net v(x, t, z) reads an external x and t no LM forward has)
#   patches           a wall both ways against tokens (a float series is no index into a table)
# Between two non-LM heads of one kind: the same readout (readout_sig / flow_sig / patch_sig equal), host widths >= the
# target's (_r3_masks_nonlm). Every other pair of values is a wall (no edge here). Catalog: verify/spot/catalog/r3.py
# r3_head_score_lm / r3_head_embedding_lm (needs theorem readout_row_subset: no PROOFS.md statement; the construction
# is the lm_head := any matrix setting of tied_sub_untied, read at fixed rows)
HEAD_ORDER = {"score": [("lm", "readout", True)], "embedding": [("lm", "readout", True)]}
INPUT_ORDER = {}

# ZEROING edges: a host with SOME layers of the target's kind reproduces a plain target on those
# layers alone and makes every other mixer sublayer the identity (zero its output projection; the
# residual and the FFN pass through, exact). The currency is DEPTH: only the host's layers of the
# target's kind count. emul_matrix applies them to PLAIN targets (softmax mixer, own-KV attention):
#   mixer   softmax <- hybrid   only the host's softmax-attention layers count (SSM / linear layers
#                               zeroed; the universal host runs Llama this way, tests/
#                               test_universal_elephantine.py; tests/test_composed_edges.py)
#   window  full    <- local    only the host's FULL (global) attention layers count (sliding ones zeroed)
#   cla     no      <- yes      only the host's layers that compute their own K,V count
# (slot, target value, host value) -> currency note. Not in the slot orders: they are depth rules,
# and composing them transitively (ssm -> softmax -> hybrid) would charge the wrong depth.
ZERO_EDGES = {("mixer", "softmax", "hybrid"): "depth (zero the SSM / linear layers)",
              ("window", "full", "local"): "depth (zero the sliding-window layers)",
              ("crosslayer_kv", "no", "yes"): "depth (zero the KV-sharing layers)"}
# looped / recurrent-depth: a block of b layers applied K times (weight-TIED). It is
# a RESTRICTION of an untied stack -- an untied model emulates it EXACTLY by unrolling
# the K steps into K*b physical layers with tied weights (free in params, paid in
# DEPTH). The reverse (looped emulates a genuinely D-deep untied model) needs
# per-iteration conditioning + width with K*b>=D iterations -- a params<->serial-compute
# trade; at the unbounded-loop limit a fixed block is Turing-complete (Universal
# Transformer / Looped-Transformers-as-Programmable-Computers). untied is the top.
LOOP_ORDER = {"looped": [("untied", "depth", True)]}         # untied is the more general (top)

TOP = {"norm": "AffineNorm", "pos": "pos_top", "kv": "MHA", "ffn": "moe_shared",
       "act": "act_top", "block": "sequential", "window": "full", "qknorm": "no", "softcap": "no",
       "mixer": "softmax", "sink": "no", "crosslayer_kv": "no", "mtp": "no", "gen": "ar",
       "loop": "untied", "head": "lm", "input": "tokens"}

SLOTS = [("norm", NORM_ORDER), ("pos", POS_ORDER), ("kv", KV_ORDER),
         ("ffn", FFN_ORDER), ("act", ACT_ORDER), ("block", BLOCK_ORDER),
         ("window", WINDOW_ORDER), ("qknorm", QKNORM_ORDER), ("softcap", SOFTCAP_ORDER),
         ("mixer", MIXER_ORDER), ("sink", SINK_ORDER), ("crosslayer_kv", CLA_ORDER),
         ("mtp", MTP_ORDER), ("gen", GEN_ORDER), ("loop", LOOP_ORDER),
         ("head", HEAD_ORDER), ("input", INPUT_ORDER)]        # (round 3, nonlm_heads: 17 slots, was 15)
# a point built by hand without the round-3 readout slots (older tests, synthetic hosts) is an LM reading tokens
SLOT_DEFAULT = {"head": "lm", "input": "tokens"}

GATED = {"llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
         "phi", "phi3", "phimoe", "gemma", "gemma2", "gemma3", "olmo", "olmo2", "olmoe",
         "deepseek", "deepseek_v2", "deepseek_v3", "starcoder2", "cohere", "cohere2",
         "stablelm", "exaone", "internlm2", "internlm3", "baichuan", "yi", "minicpm",
         "minicpm3", "granite", "granitemoe", "glm", "glm4", "chatglm",
         "dbrx", "arctic", "jamba", "zamba2", "solar", "mimo", "smollm3", "helium",
         "qwen2_5_vl", "ernie4_5", "dots1", "hunyuan",
         "internlm",            # InternLM-1 (remote): Llama body, RoPE from a nested `rotary` dict, biased q / k / v / o
         "openelm"}             # OpenELM (remote): per-layer GQA / FFN widths, SwiGLU, RMSNorm, RoPE, per-head qk-norm
VANILLA = {"gpt2", "gpt_neox", "bloom", "opt", "falcon", "mpt", "gptj", "gpt_bigcode",
           "starcoder", "codegen", "xglm", "gpt_neo", "persimmon", "mamba", "rwkv",
           "nemotron",           # Nemotron-4 / Minitron: up -> relu2 -> down, no gate (NemotronMLP)
           "biogpt", "ctrl", "openai-gpt"}
LN_FAMILIES = {"gpt2", "gpt_neox", "bloom", "opt", "falcon", "mpt", "gptj",
               "gpt_bigcode", "starcoder", "codegen", "xglm", "gpt_neo",
               "nemotron",           # NemotronLayerNorm1P (config key norm_eps)
               "biogpt", "ctrl", "openai-gpt"}
LEARNED_ABS = {"gpt2", "opt", "gpt_bigcode", "gpt_neo",
               "biogpt",             # BioGptLearnedPositionalEmbedding (offset 2)
               "openai-gpt"}         # positions_embed
# FIXED sinusoidal position tables added to the embeddings (XGLMSinusoidalPositionalEmbedding: a non-persistent
# buffer, offset 2; CTRL positional_encoding): no learned table, so not learned_abs. Its own pos value "sinusoidal"
# with NO edge for now (a learned-abs host can hold the table, but that edge is not built here: round-2 note).
SINUSOIDAL_TYPES = {"xglm", "ctrl"}
SSM_TYPES = {"mamba", "mamba2", "falcon_mamba", "rwkv", "rwkv6", "rwkv7"}
LINEAR_TYPES = {"retnet", "based", "gla", "rwkv5"}
# round 3, batch 4a (fla_num_heads_unplaced): fla DeltaNet (DeltaNetForCausalLM: every layer a delta-rule mixer,
# modeling_delta_net.py:44-55) -- it fell to "softmax" and was dropped (no num_attention_heads)
LINEAR_TYPES = _R3Set(LINEAR_TYPES, add={"fla_num_heads_unplaced": {"delta_net"}})
FLA_ARCHS = frozenset({"TransformerForCausalLM", "DeltaNetForCausalLM", "GLAForCausalLM"})   # (gated on the class)
HYBRID_TYPES = {"jamba", "zamba", "zamba2", "nemotron_h", "falcon_h1", "granitemoehybrid",
                "plamo2", "bamba", "hymba", "minimax", "minimax_text_01", "samba", "phi4flash",
                "kimi_linear",   # Kimi Delta Attention: linear + periodic full attention
                "minimax_m1",    # lightning (linear) attention + every 8th layer softmax (attn_type_list)
                # gated short conv / RG-LRU layers interleaved with (GQA / local MQA) softmax layers
                "lfm2", "lfm2_moe", "lfm3", "recurrent_gemma", "recurrentgemma",
                # Ling / Ring 2.5 - 3.x: linear-attention layers with an MLA layer every layer_group_size-th
                # (modeling_bailing_moe_v2_5 / _v3 layer rule); MiniCPM-SALA: mixer_types (minicpm4 sparse softmax
                # + lightning-attn layers)
                "bailing_hybrid", "minicpm_sala"}
# Nemotron-H schema (hybrid_override_pattern / layers_block_type, mlp_hidden_act): every norm is
# NemotronHRMSNorm despite the `layer_norm_epsilon` key, and the dense MLP is up -> relu2 -> down
NEMOTRON_H_TYPES = {"nemotron_h", "nemotron_h_puzzle", "ceno"}
# Qwen v1 (QWenLMHeadModel, remote modeling_qwen.py): ln_1 / ln_2 / ln_f are its own RMSNorm class
# (x * rsqrt(mean(x^2) + eps) * weight) built with eps=config.layer_norm_epsilon; the key name
# is not the norm type.
RMS_LNEPS_TYPES = {"qwen"}
RMS_LNEPS_ARCHS = {"QWenLMHeadModel"}
# The norm class the modeling code actually builds, where the config's eps key name says otherwise
# (each read from the model's code, 2026-11-11). Checked before any config key.
NORM_BY_CODE = {
    "qwen": "RMSNorm",              # modeling_qwen.py RMSNorm(eps=layer_norm_epsilon)
    "mamba": "RMSNorm",             # MambaRMSNorm(eps=layer_norm_epsilon)
    "falcon_mamba": "RMSNorm",      # FalconMambaRMSNorm(eps=layer_norm_epsilon)
    "exaone": "RMSNorm",            # EXAONE-3.x modeling_exaone.py ExaoneRMSNorm(eps=layer_norm_epsilon)
    "mobilint-exaone": "RMSNorm",   # proxy_exaone.py: an NPU port of the same EXAONE modeling code
    "nanochat_gpt": "RMSNorm",      # F.rms_norm, no weight
    "olmo": "LayerNorm",            # OlmoLayerNorm = F.layer_norm without affine (even with rms_norm_eps)
    "orion": "LayerNorm",           # modeling_orion.py nn.LayerNorm(eps=rms_norm_eps)
    "stablelm_epoch": "LayerNorm",  # nn.LayerNorm(eps=norm_eps)
    "pharia-v1": "LayerNorm",       # nn.LayerNorm
    "cohere": "LayerNorm",          # CohereLayerNorm: mean-centred, weight only
    "cohere2": "LayerNorm",         # Cohere2LayerNorm: same
    "persimmon": "LayerNorm",       # nn.LayerNorm(eps=layer_norm_eps)
    "starcoder2": "LayerNorm",      # nn.LayerNorm(eps=norm_epsilon)
    "rwkv7": "LayerNorm",           # fla RWKV7: nn.LayerNorm / fused LayerNorm
    "phimoe": "LayerNorm",          # PhiMoE: nn.LayerNorm(eps=rms_norm_eps, elementwise_affine=True)
    "hf_olmo": "LayerNorm",         # OLMo-1 remote code (layer_norm_type "default"): the same non-affine LN as olmo
}
NORM_BY_CODE = _R3Dict(NORM_BY_CODE, add={
    # round 3 (nanochat_port_unlisted): the renamed nanochat ports' stream norms are weight-free F.rms_norm (pit
    # modeling_pit.py:173 / :177 / :371, gaggle modeling_gaggle.py:36, sn38 modeling_nanochrono.py:12-13): RMSNorm,
    # p5's default already, for the record
    "nanochat_port_unlisted": {t: "RMSNorm" for t in ("gaggle", "pit", "sn38-nanochrono")},
    # round 3 (norm_fallback_rmsnorm_wrong): ParlerTTSDecoderLayer's self_attn / encoder_attn / final layer norms and the
    # decoder's final norm are nn.LayerNorm (affine, eps 1e-5; modeling_parler_tts.py:961 / :978 / :981 / :1373), the
    # MusicGen decoder layer the same, sundial's norm1 / norm2 / final norm too (modeling_sundial.py:190-191, :262; live:
    # sundial stays placed, nonlm_kept -- moot only with nonlm_kept off, batch 1's not_lm skip). With no eps key the
    # "modern" fallback gave them RMSNorm
    "norm_fallback_rmsnorm_wrong": {t: "LayerNorm" for t in (
        "parler_tts_decoder", "musicgen", "musicgen_decoder", "musicgen_melody", "musicgen_melody_decoder", "sundial")}})
RELU2_NAMES = ("relu2", "squared_relu", "sqrelu", "relu_squared")    # squared-ReLU activation strings
# FFN activation fixed in code when the config carries no activation key: squared ReLU
RAW_ACT_BY_CODE = {"nanochat_gpt": "relu2",    # down(relu(up x)^2)
                   "rwkv": "relu2", "rwkv5": "relu2",   # channel mix: square(relu(key x)), sigmoid receptance
                   "ctrl": "relu"}                       # point_wise_feed_forward_network: Linear -> ReLU -> Linear
# round 3 (nanochat_port_unlisted): three renamed nanochat ports with no activation key run up -> relu^2 -> down
# (gaggle modeling_gaggle.py:116-125, pit modeling_pit.py:144-152, sn38-nanochrono modeling_nanochrono.py:86-93)
RAW_ACT_BY_CODE = _R3Dict(RAW_ACT_BY_CODE, add={
    "nanochat_port_unlisted": {t: "relu2" for t in ("gaggle", "pit", "sn38-nanochrono")}})
# ungated FFN (up -> act -> down) in code although the model_type is otherwise in GATED / unknown
UNGATED_BY_CODE = {"phi", "bunny-phi",          # PhiMLP fc1 -> act -> fc2
                   "progen",                    # ProGenMLP fc_in -> act -> fc_out
                   "starcoder2",                # Starcoder2MLP c_fc -> act -> c_proj
                   "gpt_neox_japanese",         # dense_h_to_4h -> act -> dense_4h_to_h
                   "gpt_neox_nope",             # stock GPTNeoXForCausalLM, rotary off
                   "pharia-v1",                 # up_proj -> act -> down_proj
                   "nanochat_gpt",              # up -> relu^2 -> down
                   "rwkv7",                     # RWKV7FeedForward key -> sqrelu -> value
                   "apertus",                   # ApertusMLP down(xIELU(up x))
                   "arcee"}                     # ArceeMLP down(relu2(up x))
UNGATED_BY_CODE = _R3Set(UNGATED_BY_CODE, add={
    # round 3 (nanochat_port_unlisted): the nanochat ports' c_fc / up_proj -> relu^2 -> c_proj / down_proj, no gate
    "nanochat_port_unlisted": {"gaggle", "pit", "sn38-nanochrono"},
    # round 3 (ungated_ffn_type_unlisted): ParlerTTSDecoderLayer fc1 -> act -> fc2, bias-free (modeling_parler_tts.py
    # :979-980, :1060-1062), MusicgenDecoderLayer the same; p5 gave them the modern "glu" default
    "ungated_ffn_type_unlisted": {"parler_tts_decoder", "musicgen", "musicgen_decoder", "musicgen_melody",
                                  "musicgen_melody_decoder"},
    # round 3, the zoo's params triage (zoo_apertus1p5_ungated): Apertus-1.5 (gated repo, no modelling code) runs the
    # stock Apertus layer, ApertusMLP down(xIELU(up x)) with no gate (tf 4.57 modeling_apertus.py:43-54); the Hub's BF16
    # count, 8,609,083,456, is exactly the ungated LM (a gate would add 32 x 4096 x 21504 = 2.8B)
    "zoo_apertus1p5_ungated": {"apertus1p5_text", "apertus1p5"}})
# attention and FFN read the SAME normed input and are summed into the residual (code-checked). Falcon is
# parallel only when parallel_attn (class default True) or new_decoder_architecture: Falcon-RW-1B is sequential
# (FalconDecoderLayer), so it is read per config in classify, not listed here.
PARALLEL_BY_CODE = {"gptj", "codegen", "progen", "phi", "bunny-phi", "cohere", "cohere2"}
# round 3 (parallel_by_code_unlisted): Bitsy500M's BitsyBlock returns x + attn(attn_norm(x)) + mlp(mlp_norm(x))
# (modeling_bitsy.py:81-95): both sublayers read the block input under their own norms (GPT-NeoX
# use_parallel_residual), fixed in code with no key
PARALLEL_BY_CODE = _R3Set(PARALLEL_BY_CODE, add={"parallel_by_code_unlisted": {"bitsy_parallel"}})


# configs whose intermediate_size is the FUSED gate+up width (code-checked): the GLU is half of it
FUSED_GLU_HALF = {"qwen",     # modeling_qwen.py: w1, w2 = Linear(h, intermediate_size // 2)
                  "molmo"}    # modeling_molmo.py: ff_proj -> SwiGLU chunk(2) (output_multiplier 0.5)
# round 3, batch 3b (ffn_width_keys_unread): RecurrentGemma's MLP is intermediate_size // 2 wide (gate / up / down,
# tf5 modeling_recurrent_gemma.py:488-491)
FUSED_GLU_HALF = _R3Set(FUSED_GLU_HALF, add={"ffn_width_keys_unread": {"recurrent_gemma", "recurrentgemma"}})
# round 3, batch 3b (hadamard_mlp_unlisted): needle's every FFN is HadamardMLP (cactus architecture.py:298-330, :368-369):
# z = pad(x, n), n = next_pow2(d_model); P1 (A1 kron B1)(d1 z) -> silu(d2 cond(x) z + b2), cond = 1 + softmax(x V) U ->
# P2 (A2 kron B2) -> (A3 kron B3)(d3 z) -> (d4 z)[:d_model]: one ungated SiLU at width n behind Kronecker-factored maps and
# a data-dependent input scale. Its own ffn value, hosted only by its own code at equal n (dims hada_n)
HADAMARD_FFN_TYPES = _R3Set(set(), add={"hadamard_mlp_unlisted": {"needle"}})
# round 3 (mamba_ffn_none): the model types whose block has no FFN at all (FFN_ORDER "none"; ffn widths 0). Checked by
# code: transformers MambaBlock (modeling_mamba.py:478-503: norm, mixer, residual), FalconMambaBlock
# (modeling_falcon_mamba.py:543-), Mamba2Block (modeling_mamba2.py:683-). Not RWKV (RwkvBlock's feed_forward channel
# mix, modeling_rwkv.py:360; RWKV-5 / -7 likewise), not fla's GLA / DeltaNet blocks (an MLP after the mixer)
FFN_NONE_TYPES = _R3Set(set(), add={"mamba_ffn_none": {"mamba", "falcon_mamba", "mamba2"}})


def _moe_like(d):
    return bool(d.get("num_experts") or d.get("moe_num_experts") or d.get("n_routed_experts"))


# --- config-class defaults: a key ABSENT from a saved config is the config class's default ------------------------
# (save_pretrained writes the diff against the class). Read from transformers' own config classes
# (CONFIG_MAPPING[model_type]()), not from hand lists. transformers-5-only and remote-code types are not in the
# installed mapping; their defaults below are code-checked (transformers 5.18 configuration_*.py; the repos'
# remote configuration / modeling files).
_CLASS_DEFAULT_FALLBACK = {
    # transformers-5-only MoE types whose router reads norm_topk_prob (class default True): Qwen4-exp
    "norm_topk_prob": {t: True for t in ("qwen4_exp", "qwen4_exp_text")},
    "tie_word_embeddings": {
        **{t: True for t in (
            "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "diffusion_gemma",
            "diffusion_gemma_text", "lfm2_moe", "minicpm3", "cohere2_moe", "granite_swa", "kimi_k25", "youtu", "zaya",
            "neomme", "ernie4_5_vl_moe", "ernie4_5_vl_moe_text", "hunyuan_vl", "hunyuan_vl_text", "paddleocr_vl",
            "paddleocr_vl_text", "qwen2_5_omni_text",
            "plamo2", "plamo2vl")},                # PLaMo-2 remote PlamoConfig: tie_word_embeddings=True
        # transformers-5-only LM types whose class default is False (a wrapper's True must not leak into them)
        **{t: False for t in (
            "afmoe", "axk1", "axk2", "cwm", "deepseek_v32", "deepseek_v4", "exaone4_5", "exaone_moe", "glm4_moe_lite",
            "glm5_next", "glm5_next_text", "glm_moe_dsa", "granitemoe_swa", "hy_v3", "hy_v4", "hyperclovax", "jais2",
            "kimi_linear", "laguna", "mellum", "mimo_v2_flash", "minimax_m2", "minimax_m3_vl", "minimax_m3_vl_text",
            "ministral3", "mistral4", "muse_glimmer_text", "nanochat", "nemotron_h", "olmo_hybrid", "qwen3_5",
            "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text", "solar_open",
            "step3p5")}},
    "partial_rotary_factor": {"chatglm": 0.5},     # ChatGLM / GLM-4-9B remote: RotaryEmbedding(kv_channels // 2)
}
_CLASS_DEFAULT_FALLBACK.update({
    "partial_rotary_factor": {"chatglm": 0.5},     # ChatGLM / GLM-4-9B remote: RotaryEmbedding(kv_channels // 2)
})
# round 3 (partial_rope_keys_unread): ZAYA1-VL's config class defaults partial_rotary_factor 0.5 and folds it into
# rope_parameters (configuration_zaya1_vl.py:99, :157, :170; the snapshot's rope_pct is stored but unread); Grug-MoE's
# _apply_half_rope rotates head_dim // 2 by code (skyrl_train_models_grug_moe.py:680-700, theta^(-i/32) on 64 of 128)
_CLASS_DEFAULT_FALLBACK["partial_rotary_factor"] = _R3Dict(_CLASS_DEFAULT_FALLBACK["partial_rotary_factor"], add={
    "partial_rope_keys_unread": {"zaya1_vl": 0.5, "grug_moe": 0.5}})
# round 3 (tie_default_unread): remote config classes defaulting tie_word_embeddings=True whose saved configs omit
# the key (lm_head = E): MiniCPM (configuration_minicpm.py:128, typeless configs keyed by architecture in _tied),
# Phi-3-small (PretrainedConfig default True, modeling_phi3_small.py:900 _tied_weights_keys), Cyrah
# (configuration_cyrah.py:281), Nexus (configuration_nexus.py:36), Kanana-2-tiny (configuration_kanana2_tiny.py:59),
# TinyMixtral (configuration_tinymixtral.py:27), Bitsy (configuration_bitsy.py:13), Gemma3-MoE (Gemma3TextConfig:
# configuration_gemma3.py:186)
_CLASS_DEFAULT_FALLBACK["tie_word_embeddings"] = _R3Dict(_CLASS_DEFAULT_FALLBACK["tie_word_embeddings"], add={
    "tie_default_unread": {t: True for t in ("minicpm", "cyrah", "nexus", "kanana2_tiny", "phi3small", "tinymixtral",
                                             "bitsy_parallel", "gemma3_moe")}})
# round 3 (config_default_unread): LLaDA2-MoE's class default use_qk_norm=True (configuration_llada2_moe.py:19, :57;
# modeling_llada2_moe.py:448 / :496 build and apply query_layernorm / key_layernorm under it; the 2.0-mini-preview
# applies them unconditionally, :396 / :431); Muse-Glimmer's final_logit_softcapping = 20.0
# (configuration_muse_glimmer.py:141; modeling_muse_glimmer.py:1188-1193 applies T tanh(l / T) unconditionally)
_CLASS_DEFAULT_FALLBACK["use_qk_norm"] = _R3Dict({}, add={"config_default_unread": {"llada2_moe": True}})
_CLASS_DEFAULT_FALLBACK["final_logit_softcapping"] = _R3Dict({}, add={
    "config_default_unread": {"muse_glimmer": 20.0, "muse_glimmer_text": 20.0}})


def _class_config(mt):
    if mt not in _CLASS_CFG:
        cfg = None
        try:
            from transformers import CONFIG_MAPPING
            if mt in CONFIG_MAPPING:
                import logging
                lg = logging.getLogger("transformers")
                lvl = lg.level
                lg.setLevel(logging.ERROR)
                try:
                    cfg = CONFIG_MAPPING[mt]()
                finally:
                    lg.setLevel(lvl)
        except Exception:
            cfg = None
        _CLASS_CFG[mt] = cfg
    return _CLASS_CFG[mt]


_CLASS_CFG = {}


def _class_default(mt, key):
    """The default of `key` in model type mt's config class (None if unknown)."""
    mt = str(mt or "").lower()
    cfg = _class_config(mt) if mt else None
    if cfg is not None and hasattr(cfg, key):
        return getattr(cfg, key)
    return _CLASS_DEFAULT_FALLBACK.get(key, {}).get(mt)


def _cfg_value(d, key, outer=None):
    """The value the model runs with: the saved key, else the outer wrapper's, else the class default of the LM's
    model type, else the outer type's."""
    if key in d and d[key] is not None:
        return d[key]
    if outer and key in outer and outer[key] is not None:
        return outer[key]
    v = _class_default(d.get("model_type"), key)
    if v is None and outer:
        v = _class_default(outer.get("model_type"), key)
    return v


def raw_act(d):
    """The FFN activation string the model's code applies: the config key its MLP reads (ACT_KEY_BY_CODE where
    the family reads one key and ignores another), else the usual keys, else the config class's default for
    that key (an absent key is the class default), else a code-fixed activation."""
    mt = str(d.get("model_type") or "").lower()
    if mt in ACT_KEY_BY_CODE:
        k = ACT_KEY_BY_CODE[mt]
        a = d.get(k) or (_class_default(mt, k) if USE_ACT_CODE else None)
    else:
        a = (d.get("mlp_hidden_act") or d.get("hidden_activation") or d.get("hidden_act")
             or d.get("activation_function") or (d.get("activation") if mt == "falcon" else None))
        if not a and USE_ACT_CODE:
            a = RAW_ACT_BY_CODE.get(mt) or next((_class_default(mt, k) for k in ("hidden_act", "activation_function",
                                                                                   "hidden_activation", "activation")
                                                 if isinstance(_class_default(mt, k), str)), "")
    return str(a or "").lower()


# the activation key the family's MLP actually reads (code-checked, transformers 4.57): GemmaMLP applies
# ACT2FN[config.hidden_act] (a Gemma-1 config's hidden_activation is ignored); Gemma-2 / 3 / 3n / 4 / RecurrentGemma /
# VaultGemma / DiffusionGemma read hidden_activation (a stray hidden_act is ignored)
ACT_KEY_BY_CODE = {"gemma": "hidden_act", **{t: "hidden_activation" for t in (
    "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text", "gemma4_unified_text",
    "recurrent_gemma", "vaultgemma", "diffusion_gemma", "diffusion_gemma_text")}}
# GeLU: ACT2FN["gelu"] / nn.GELU() is the exact erf form; these strings run the tanh approximation
# 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). The two are distinct act values: "GeLU" (erf), "GeLU_tanh".
GELU_TANH_NAMES = ("gelu_new", "gelu_fast", "gelu_pytorch_tanh", "gelu_tanh", "gelu_accurate", "gelu_python_tanh",
                   "approx_gelu", "gelu_approx")
# families whose CODE maps the config string differently from ACT2FN (code-checked, transformers 4.57):
#   openai-gpt  ACT_FNS["gelu"] = gelu_new (tanh);  bloom  bloom_gelu_forward (tanh, no key);
#   mpt  nn.GELU(approximate="none") (erf, no key);  falcon  get_activation(config.activation), default "gelu" (erf)
GELU_FORM_BY_CODE = {"openai-gpt": "GeLU_tanh", "bloom": "GeLU_tanh", "mpt": "GeLU"}


USE_ACT_CODE = True
# per-layer type names of a RECURRENT (non-softmax) mixer layer
_REC_LAYER_NAMES = ("mamba", "linear", "conv", "jet", "gka", "gdn", "delta", "kda", "rwkv", "ssm",
                    "retention", "lightning")
# learned softmax sinks fixed in code (no config key): gpt-oss; Flint (softmax_offset on every layer); Granite-swash
# (granite_swa / granitemoe_swa: a learnable per-head sink, sigmoid(lse - sink) = an extra denominator logit); Hy4;
# DeepSeek-V4 (s_aux; V4.1 assumed the same: its code is not public)
SINK_TYPES = {"gpt_oss", "gpt_oss_puzzle", "flint", "granite_swa", "granitemoe_swa", "hy_v4",
              "deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4"}
# config switches for learned sinks: MiMo-V2 add_swa / add_full_attention_sink_bias, IQuest-Q1 enable_sink_attention
_SINK_KEYS = ("attention_sinks", "use_attention_sink", "add_swa_attention_sink_bias", "add_full_attention_sink_bias",
              "enable_sink_attention", "softmax_offset")
DIFFUSION_TYPES = {"llada", "llada2", "llada_moe", "dream", "mmada", "diffuqwen35",   # bidirectional denoising LMs
                   "a2d-qwen3", "diffurefill"}           # dLLM A2D (padding-only bidirectional mask); DiffuRefill
# BLOCK-causal denoising LMs (bidirectional inside a block, causal across blocks): SDAR. gen = "block" (GEN_ORDER: one
# class with "diffusion" under the mask currency, AR below both; the universal host builds no block mask, an open host
# branch). Also any non-drafter config with block_size + mask_token_id.
BLOCK_DIFFUSION_TYPES = {"sdar", "sdar_moe"}
CLA_TYPES = {"yoco", "zamba2"}                          # cross-layer KV sharing (rare / heuristic)
# weight-tied / recurrent-depth stacks (one block looped K times). ALBERT is the
# textbook case (an encoder, so usually filtered); the in-scope decoder LMs are the
# latent-recurrent-depth models (Huginn/Raven, CoTFormer, Universal-Transformer).
LOOPED_TYPES = {"albert", "huginn", "huginn_raven", "raven", "universal_transformer",
                "cotformer", "recurrent_transformer"}
SKIP_TYPES = {"clip", "dia", "gemma3n", "granite_speech", "videomae", "whisper",
              "bert", "roberta", "t5", "vit", "wav2vec2", "siglip",
              # not a plain causal decoder (literature review, round 2): LSH + chunked local attention with axial
              # positions and reversible residuals; two-stream permutation LM; segment recurrence with a relative
              # sinusoid
              "reformer", "xlnet", "transfo-xl",
              # kept out until they can be classified faithfully: BD3-LM (a DiT with adaLN that reads x_t and x_0
              # concatenated under a block mask; code not public in the snapshot), LangFlow (continuous
              # embedding-space flow with a time input and self-conditioning)
              "bd3lm", "langflow"}
# round 3 (remote_code_not_config_family): wrappers whose remote code replaces the family the config names --
# GLM-5.3-SuperExpert (configuration_glm_superexpert.py:50-51 get_text_config returns self; modeling: plain MHA with no
# RoPE or causal mask, a dense soft MoE over every expert, packed weights matching no GLM-5.3 layout) and Quillan-Ronin
# (modeling_quillan_oni.py:11-27 wraps a native LayerNorm / BitLinear / soft all-expert mixture; weights_status
# checkpoint_pending): classifying them from the config describes no real checkpoint
SKIP_TYPES = _R3Set(SKIP_TYPES, add={"remote_code_not_config_family": {"glm_superexpert", "quillan_ronin_oni"}})
# Speculative drafters / assistants are NOT standalone LMs: their input is the target model's hidden states or KV
# (DFlash / DSpark target_layer_ids, EAGLE-3 fused features, Gemma-4 assistants reading the backbone's KV through
# shared_kv_states, centroid-masked logits). A config carrying one of these keys (or an *_assistant model type) is
# skipped. (DeepSeek-V4's dspark_* keys describe a draft head ATTACHED to a full model, so they do not skip.)
DRAFTER_KEYS = ("dflash_config", "num_target_layers", "draft_vocab_size", "eagle_config", "backbone_hidden_size")


def _is_drafter(outer, d):
    mts = {str(outer.get("model_type") or "").lower(), str(d.get("model_type") or "").lower()}
    return any(k in outer or k in d for k in DRAFTER_KEYS) or any(m.endswith("_assistant") for m in mts)


def skipped(c):
    """Why classify drops a config before reading it (None if it does not): a non-decoder model type (the
    outer wrapper's or the LM's: Gemma-3n's outer gemma3n wraps a gemma3n_text) or a drafter / assistant."""
    d = _lm_config(c)
    if not isinstance(d, dict):
        return None
    mts = {str(c.get("model_type") or "").lower(), str(d.get("model_type") or "").lower()}
    if mts & SKIP_TYPES:
        return "non_decoder"
    if _is_drafter(c, d):
        return "drafter"
    # (round 3, nonlm_kept -- a design decision: the two score-head rerankers and sundial stay placed, walled from the
    # LMs by the features score_head / ts_patch (_feats, FEATURES) and carried by the universal host's join; with
    # nonlm_kept in R3_OFF batch 1's skip comes back)
    if _r3("not_a_causal_lm") and not _r3("nonlm_kept") and _not_lm(c, d):
        return "not_lm"
    return None


def _not_lm(c, d):
    """Round 3 (not_a_causal_lm): a decoder whose head emits no next-token logits -- "score_head" (the MiniCPM
    layer-wise reranker: head_type not raw / complex builds LayerWiseHead(hidden, 1) per exit layer,
    modeling_minicpm_reranker.py:1134-1185; the Gemma-2 cost-wise reranker with layer_wise: CostWiseHead(hidden, 1),
    gemma_model.py:485-511) or "continuous_io" (Sundial: float value patches in, a flow-matching sample out,
    modeling_sundial.py:28-45, :403), else None. (The two rerankers' heads are a different module, a feature switch.)"""
    mt = str(d.get("model_type") or "").lower()
    archs = set(c.get("architectures") or ()) | set(d.get("architectures") or ())
    if mt == "minicpm" and ("head_type" in d or "LayerWiseMiniCPMForCausalLM" in archs):
        if str(d.get("head_type", "simple")).lower() not in ("raw", "complex"):
            return "score_head"
    lw = d.get("layer_wise", c.get("layer_wise"))
    if mt == "cost_wise_gemma" and (lw is True or str(lw).lower() == "true"):
        return "score_head"
    if mt == "sundial" or "SundialForPrediction" in archs:
        return "continuous_io"
    return None


# round 3 (nonlm_heads): the sentence-embedding models of the snapshot. Their configs are plain *ForCausalLM configs
# (Qwen3-Embedding's is a stock Qwen3ForCausalLM), so the readout is in the repo's sentence-transformers sidecar
# (modules.json, 1_Pooling/config.json, 2_Dense*/config.json) and remote code, read as text on branch host-nonlm
# (elephantine/loader_coverage.py READOUTS, copied here): the pooling, L2 Normalize, and a Dense (in, out, bias). The
# repo id rides along under "_repo_id" (iter_json_configs), as for FAKE_QUANT_REPOS. Every one pools by the mean or
# normalises: none is a fixed read of an LM's output (HEAD_ORDER's readout rule).
_EMB_LAST_L2 = {"pool": "last", "l2": True}
EMBED_READOUTS = {
    **{r: _EMB_LAST_L2 for r in (
        "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B",
        "boboliu/Qwen3-Embedding-4B-W4A16-G128", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "mlx-community/Qwen3-Embedding-4B-4bit-DWQ", "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
        "mlx-community/Qwen3-Embedding-8B-mxfp8", "jinaai/jina-code-embeddings-0.5b", "jinaai/jina-code-embeddings-1.5b",
        # (bidirectional bodies: gen diffusion from the config's is_causal false)
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct",
        # VLM2Vec (its own GitHub code): the last real token's hidden state, normalised
        "TIGER-Lab/VLM2Vec-Full")},
    # mean pooling, then 2_Dense 1536 -> 1024 with a bias, no Normalize (bidirectional modeling_qwen.py Qwen2Model)
    "NovaSearch/stella_en_1.5B_v5": {"pool": "mean", "l2": False, "dense": (1536, 1024, True)},
    "sdadas/stella-pl-retrieval-8k": {"pool": "mean", "l2": False, "dense": (1536, 1024, True)},
    # Qwen3BidirectionalModel: a per-token Linear(hidden, num_labels) without a bias, mean pooling, Normalize
    "MongoDB/voyage-4-nano": {"pool": "mean", "l2": True, "dense": (1024, 2048, False)},
}
# readouts that are a fixed read of an LM's output (a position subset, no reduction over positions, no L2)
FIXED_READS = frozenset({"last", "first"})


def _readout(d, outer=None):
    """Round 3 (nonlm_heads): (head, input, info) of an LM config d (outer = its wrapper): head lm | score | embedding |
    value | flow, input tokens | patches; info {readout_sig, readout_w, readout_ff, readout_bias, flow_sig, patch_sig,
    exits} (dims_of's R3_DIMS_NONLM; neutral for an LM)."""
    outer = outer if isinstance(outer, dict) else {}
    info = {"readout_sig": "", "readout_w": 0, "readout_ff": 0, "readout_bias": 0, "flow_sig": "", "patch_sig": "",
            "exits": 0}
    if not _r3("nonlm_heads"):
        return "lm", "tokens", info
    g = lambda k: _num(d.get(k, outer.get(k)))
    hidden = g("hidden_size")
    rid = d.get("_repo_id") or outer.get("_repo_id")
    if rid in EMBED_READOUTS:
        e = EMBED_READOUTS[rid]
        dense = e.get("dense")
        info.update(readout_sig=e["pool"] + ("+l2" if e["l2"] else ""), readout_w=dense[1] if dense else hidden,
                    readout_bias=int(bool(dense and dense[2])))
        return "embedding", "tokens", info
    # (the score / flow detection is batch 1's not_a_causal_lm finding: with it in R3_OFF they read as LMs again)
    nl = _not_lm(outer or d, d) if _r3("not_a_causal_lm") else None
    if nl == "score_head":
        # one Linear(hidden, 1, bias=False) per exit layer in range(start_layer, L + 1, layer_sep)
        # (modeling_minicpm_reranker.py:1134-1142 / :1144-1160; gemma_model.py:485-511); the default call reads one
        L = g("num_hidden_layers")
        s0, sep = g("start_layer"), g("layer_sep") or 1
        info.update(readout_w=1, exits=len(range(s0, L + 1, sep)) if L else 0)
        return "score", "tokens", info
    if nl == "continuous_io":
        # sundial: SundialPatchEmbedding of input_token_len-value patches through an intermediate_size MLP
        # (modeling_sundial.py:28-45); FlowLoss's SimpleMLPAdaLN(output_token_lens[-1], hidden, depth flow_loss_depth)
        # (flow_loss.py:40-49, modeling_sundial.py:403)
        outs = d.get("output_token_lens") or [0]
        info.update(readout_w=int(_num(max(outs) if isinstance(outs, list) else outs)),
                    flow_sig=f"{int(_num(max(outs) if isinstance(outs, list) else outs))}/{int(hidden)}/"
                             f"{int(g('flow_loss_depth'))}",
                    patch_sig=f"{int(g('input_token_len'))}/{int(g('intermediate_size'))}")
        return "flow", "patches", info
    am = d.get("auto_map") or outer.get("auto_map") or {}
    if (d.get("vhead_layer_type") or outer.get("vhead_layer_type")
            or (isinstance(am, dict) and am.get("AutoModel") == "model.AutoModelForCausalLMWithValueHead")):
        # GRM's ValueHead (model.py:9-60): Linear(hidden, vhead_num_neurons) -> ReLU -> Linear(., 1) for "mlp" (nn.Linear
        # with its bias), a single Linear(hidden, 1) for "linear"; read at the last real token
        mlp = str(d.get("vhead_layer_type", outer.get("vhead_layer_type", "mlp"))).lower() == "mlp"
        info.update(readout_w=1, readout_ff=int(g("vhead_num_neurons") or 1024) if mlp else 0, readout_bias=1)
        return "value", "tokens", info
    return "lm", "tokens", info


def _looped(d):
    """Weight-tied / recurrent-depth: one block of layers applied K times."""
    if d.get("model_type") in LOOPED_TYPES:
        return True
    if any(_num(d.get(k)) > 1 for k in _LOOP_COUNT_KEYS):
        return True
    mt = str(d.get("model_type") or "").lower()
    if _r3("loop_depth_unread") and ((mt == "ouro" and (_num(d.get("total_ut_steps")) or 4) > 1)
                                     or (mt == "cfrd" and _num(d.get("physical_cells")))):
        # round 3: Ouro runs the whole stack total_ut_steps times (getattr default 4, modeling_ouro.py:560, :640-652);
        # CFRD runs its physical_cells cells round-robin for inference_recurrences steps (haru model.py:519, :605-606)
        return True
    if _r3("hrm_hierarchical_loop") and mt == "hrm_text":
        return True                     # round 3: HRM's two stacks run H_cycles x (L_cycles + 1) passes
    return bool(d.get("mean_recurrence") or d.get("num_recurrence") or d.get("recurrence")
                or d.get("n_layers_in_recurrent_block") or d.get("tie_layers")
                or d.get("share_layer_weights") or d.get("reuse_layers") or d.get("layer_sharing"))


# loop / recurrence counts under family-specific names: ARLoop loop_count, Nanbeige / qwen3loop
# num_loops, IQuest-Loop loop_num, kotodama loop_times, Dory n_recurrent_loops, LFM3-RD
# recurrent_loops, Looped-Qwen loops_per_group
_LOOP_COUNT_KEYS = ("loop_count", "num_loops", "loop_num", "loop_times", "n_recurrent_loops",
                    "recurrent_loops", "loops_per_group")


def _loop_depth(d, n_layers):
    """(param_layers, eff_layers) of a looped stack whose loop is given by count + span keys.
    Which of the two the config's layer count already is differs per schema."""
    g = lambda k: _num(d.get(k))
    mt = str(d.get("model_type") or "").lower()
    if _r3("hrm_hierarchical_loop") and mt == "hrm_text":
        # round 3: HRM-Text (tf5 modular_hrm_text.py:98-120, :222-230, :348-400): two stacks L / H of P layers each,
        # H_cycles x (L_cycles + 1) stack passes; a saved num_layers_per_stack means num_hidden_layers is already
        # the unrolled count (__post_init__ rewrites it), else P is the saved num_hidden_layers
        H = g("H_cycles") if "H_cycles" in d else 2
        L = g("L_cycles") if "L_cycles" in d else 3
        P = g("num_layers_per_stack") or n_layers or 16
        if H and P:
            return 2 * P, P * H * (L + 1)
    if _r3("loop_split_depth_miscount") and d.get("enable_double_loop_split") is True and n_layers:
        # round 3: get_double_loop_split_layer_order (modeling_qwen3loop.py:37-75; Nanbeige4.2 :403-430): prefix and
        # suffix once, the middle loop_middle_layers (default n // 2) repeated (n + m) // m times (28 -> 56)
        m = g("loop_middle_layers") or n_layers // 2
        if m and not n_layers % m:
            return n_layers, n_layers + m * ((n_layers + m) // m - 1)
    if _r3("loop_depth_unread") and mt == "cfrd" and g("physical_cells"):
        # round 3: CFRD (haru model.py:519, :605-606; modeling_cfrd.py:70): inference_recurrences (default recurrences,
        # configuration_cfrd.py:72) steps over physical_cells cells round-robin, plus haru_1.1 / 2's BindingBlock
        # before the logits under use_binding_block (haru_2 model.py:622, :748-749)
        cells = g("physical_cells") or 2
        rec = g("inference_recurrences") or g("recurrences") or 6
        bind = 1 if d.get("use_binding_block") is True else 0
        return min(cells, rec) + bind, rec + bind
    if g("physical_layers"):                                  # ARLoop: num_hidden_layers is unrolled
        return g("physical_layers"), max(n_layers, g("physical_layers") * max(1, g("loop_count")))
    if g("loop_len") and g("loop_times"):                     # kotodama: num_layers is unrolled
        return max(1, n_layers - g("loop_len") * (g("loop_times") - 1)), n_layers
    span_rep = [("n_recurrent_layers", "n_recurrent_loops"),  # Dory: in + recurrent x loops + out
                ("loop_middle_layers", "num_loops")]          # qwen3loop: the middle block repeats
    for span, rep in span_rep:
        if g(span) and g(rep):
            return n_layers, n_layers + g(span) * (g(rep) - 1)
    if g("num_motifs") and g("motif_size") and g("recurrent_loops"):   # LFM3-RD: motif body repeats
        if _r3("loop_depth_unread"):
            # round 3: one motif PHASE per recurrent step (m_idx = s % num_motifs, modeling_lfm3_rd.py:1102-1113), steps
            # below num_motifs finish the tail FFN-only (:937-947): every layer once, plus motif_size per extra step
            return n_layers, n_layers + g("motif_size") * max(0, g("recurrent_loops") - g("num_motifs"))
        return n_layers, n_layers + g("num_motifs") * g("motif_size") * (g("recurrent_loops") - 1)
    rec = next((g(k) for k in _LOOP_COUNT_KEYS if g(k) > 1), 1)   # whole stack applied rec times
    if _r3("loop_depth_unread") and mt == "ouro":
        rec = g("total_ut_steps") or 4                        # round 3: the whole stack total_ut_steps times
    return n_layers, n_layers * rec


# stock dense HF classes: MoE / head_dim fields in their config are leftovers the code never reads
# (NSFW-flash: a StableLmForCausalLM config carrying num_local_experts=8 and head_dim=256)
_DENSE_ARCHS = {"LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
                "StableLmForCausalLM", "GemmaForCausalLM", "Gemma2ForCausalLM", "Phi3ForCausalLM"}
_DERIVED_HEAD_DIM_ARCHS = {"StableLmForCausalLM"}          # head_dim = hidden_size // heads


def _stock_arch(d):
    """The stock transformers class a config runs under (None if it ships remote code)."""
    a = d.get("architectures")
    return a[0] if isinstance(a, list) and a and not d.get("auto_map") else None


def _moe_disabled(d):
    """The config carries expert counts that the model does NOT use, so it is dense:
      - an explicit off switch (Doge `is_moe: false`, `use_moe: false`) next to num_experts;
      - a Nemotron-H-style per-layer type list that names dense-FFN layers but NO MoE layer
        ('mlp' / '-' but no 'moe' / 'E'): the n_routed_experts / moe_* fields are then template
        defaults (e.g. a 4B NemotronH still carrying n_routed_experts=8 whose 42 layers are
        mamba / attention / mlp only)."""
    if d.get("is_moe") is False or d.get("use_moe") is False or _stock_arch(d) in _DENSE_ARCHS:
        return True
    L = _num(d.get("num_hidden_layers"))
    if _r3("router_feats_on_dense_model") and L and _dense_first(d) >= L:
        return True                     # round 3: every layer dense (Cyrah first_k_dense_replace = num_hidden_layers)
    lbt, pat = d.get("layers_block_type"), d.get("hybrid_override_pattern")
    if isinstance(lbt, list) and lbt:
        s = {str(x).lower() for x in lbt}
        return "mlp" in s and "moe" not in s
    if isinstance(pat, str) and pat:
        return "-" in pat and "E" not in pat
    return False


def _ffn_sublayers(d):
    """(n_moe, n_dense) FFN sub-layers when the Nemotron-H schema lists the mixer and the FFN of a
    block as SEPARATE entries (hybrid_override_pattern 'M' / '*' / 'E' / '-', layers_block_type
    'mamba' / 'attention' / 'moe' / 'mlp'), else None. Only those entries carry an FFN, so the
    whole-model FFN budget is n_moe * (experts + shared) + n_dense * dense width, not
    len(list) * ffn_total (which prices every Mamba and attention entry as a full MoE layer)."""
    if d.get("model_type") not in NEMOTRON_H_TYPES:
        return None
    pat = d.get("hybrid_override_pattern")
    if isinstance(pat, str) and pat:
        moe, dense = pat.count("E"), pat.count("-")
    elif isinstance(d.get("layers_block_type"), list):
        lbt = [str(x).lower() for x in d["layers_block_type"]]
        moe, dense = lbt.count("moe"), sum(x in ("mlp", "ffn") for x in lbt)
    else:
        return None
    return (moe, dense) if moe + dense else None


# top-k keys other than num_experts_per_tok / moe_topk / moe_top_k / moe_k, each read from the code:
#   num_experts_per_token  Kimi-Linear / Kimi-K3 (modeling_kimi.py top_k = config.num_experts_per_token)
#   top_k_experts          Gemma-4 / DiffusionGemma router (torch.topk(k=config.top_k_experts))
#   experts_top_k          Motif-3 (modeling_motif.py torch.topk(..., k=self.experts_top_k))
#   num_selects            LLaMA-MoE gate (logits.topk(num_selects))
_TOPK_ALT_KEYS = ("num_experts_per_token", "top_k_experts", "experts_top_k", "num_selects")


def _dense_mlp_beside_moe(d):
    """Gemma-4 (enable_moe_block) and DiffusionGemma: every decoder layer runs its dense MLP
    (intermediate_size) AND the routed experts on the same residual, summed -- an always-on
    shared FFN of width intermediate_size (modeling_gemma4.py Gemma4TextDecoderLayer)."""
    mt = str(d.get("model_type") or "")
    return bool(d.get("enable_moe_block")) or (mt.startswith("diffusion_gemma") and bool(d.get("num_experts")))


def _dense_first(d):
    """Leading dense (non-MoE) layers: first_k_dense_replace (DeepSeek, Ling, Kimi, ...),
    num_dense_layers (AFMoE, LFM2-MoE), n_dense_first_layers (Motif-3); round 3 (dense_first_key_unread): Kohaku's
    first_k_dense, class default 1 (modeling_kohaku.py:178-186, configuration_kohaku.py:36, 61)."""
    return (_num(d.get("first_k_dense_replace")) or _num(d.get("num_dense_layers")) or _num(d.get("n_dense_first_layers"))
            or (_num(d.get("first_k_dense", 1)) if _r3("dense_first_key_unread")
                and str(d.get("model_type") or "").lower() == "kohaku" else 0))


def _moe_layer_mask(d, n_layers):
    """Per-layer FFN kind of a MoE stack: a list of n_layers bools (True = a routed-MoE FFN, False = a dense
    MLP), read the way each family's code builds its layers (round 2, pass 3), or None (every layer MoE). The
    keys, by family (transformers 5.18 modeling_*.py unless noted):
      mlp_layer_types       'dense' / 'sparse' / 'moe' / 'hash_moe' per layer (GLM-4.7-Flash, GLM-5, Hy3 / Hy4,
                            Laguna, EXAONE-MoE, Mellum, DeepSeek-V4, Beam, Mistral-Large-4, Muse-Spark)
      moe_layer_freq list   1 = MoE, 0 = dense per layer (MiMo-V2 / V2-Flash remote code, MiniMax-M3, Naive-N0.5)
      moe_layers_enum       the MoE layer indices, comma-separated (Step-3 / Step-3.5 remote code)
      moe_layers / interleave_moe_layer_step   Llama-4: moe_layers, else range(step - 1, L, step)
      mlp_only_layers + decoder_sparse_step   Qwen-MoE family: MoE iff i not in mlp_only_layers and
                            (i + 1) % step == 0 (Qwen1.5 / 2 / 3-MoE, Qwen3-Next / VL-MoE / Omni; Flint, K2-Horizon)
      moe_layer_start_index / _end_index / _interval   ERNIE-4.5: MoE iff (i + 1) % interval == 0 and
                            start <= i <= end (end -1 = the last layer)
      expert_layer_period / _offset   Jamba: MoE iff i % period == offset
      first_k_dense_replace (+ moe_layer_freq int)   DeepSeek / Kimi / GLM / Ling / dots: MoE iff i >= k and
                            i % freq == 0; num_dense_layers (AFMoE, LFM2-MoE), n_dense_first_layers (Motif-3)"""
    L = int(n_layers or 0)
    if L <= 0:
        return None
    mlt = d.get("mlp_layer_types")
    if isinstance(mlt, list) and len(mlt) == L:
        return [str(x).lower() != "dense" for x in mlt]
    mlf = d.get("moe_layer_freq")
    if isinstance(mlf, list) and len(mlf) == L:
        return [bool(_num(x)) for x in mlf]
    enum = d.get("moe_layers_enum")
    if enum:
        idx = {int(x) for x in (enum.split(",") if isinstance(enum, str) else enum) if str(x).strip().isdigit()}
        if idx:
            return [i in idx for i in range(L)]
    mt = str(d.get("model_type") or "").lower()
    if mt.startswith("llama4"):
        ml = d.get("moe_layers")
        if isinstance(ml, list) and ml:
            return [i in set(ml) for i in range(L)]
        step = max(1, _num(d.get("interleave_moe_layer_step")) or 1)
        return [i in set(range(step - 1, L, step)) for i in range(L)]
    if d.get("mlp_only_layers") or _num(d.get("decoder_sparse_step")) > 1:
        only = set(x for x in (d.get("mlp_only_layers") or []) if isinstance(x, int))
        step = max(1, _num(d.get("decoder_sparse_step")) or 1)
        return [i not in only and (i + 1) % step == 0 for i in range(L)]
    if d.get("moe_layer_start_index") is not None or _num(d.get("moe_layer_interval")) > 1:
        start = _num(d.get("moe_layer_start_index"))
        end = d.get("moe_layer_end_index")
        end = L - 1 if end is None or _num(end) < 0 else _num(end)
        iv = max(1, _num(d.get("moe_layer_interval")) or 1)
        return [(i + 1) % iv == 0 and start <= i <= end for i in range(L)]
    if mt == "jamba" or d.get("expert_layer_period") is not None:
        per = max(1, _num(d.get("expert_layer_period")) or 2)
        off = _num(d.get("expert_layer_offset")) if d.get("expert_layer_offset") is not None else 1
        return [i % per == off for i in range(L)]
    k = _dense_first(d)
    freq = _num(d.get("moe_layer_freq")) if not isinstance(d.get("moe_layer_freq"), list) else 0
    step = _num(d.get("interleave_moe_layer_step")) if mt == "motif" else 0
    if k or freq > 1 or step > 1:
        freq = max(freq, step, 1)
        return [i >= k and i % freq == 0 for i in range(L)]
    return None


# --- MoE routing (round 2, pass 3) ---------------------------------------------------------------------------
# The renormalisation of the top-k weights, from the CODE (lit review X2): R = divided by the selected mass
# (softmax-after-top-k, norm_topk_prob = True, sigmoid / sum), P = not (softmax over all experts, then top-k).
# Fixed in code (the config's norm_topk_prob, if any, is ignored):
RENORM_FIXED = {
    **{t: "R" for t in ("mixtral", "tinymixtral", "qwen3_5_moe", "qwen3_5_moe_text", "memt_qwen3_5_moe", "qwen3_vl_moe",
                        "qwen3_vl_moe_text", "granitemoe", "granitemoeshared", "granitemoehybrid", "granitemoe_swa",
                        "jetmoe", "gpt_oss", "gpt_oss_puzzle", "gemma4", "gemma4_text", "gemma4_unified_text",
                        "diffusion_gemma", "diffusion_gemma_text", "minimax", "minimax_text_01", "minimax_m1",
                        "minimax_m2", "minimax_m3_vl", "hunyuan_v1_moe", "hunyuan_image_3_moe", "hy_v3", "afmoe",
                        "laguna", "ernie4_5_moe", "step3p7", "deepseek_v4", "deepseek_v41_text", "memt_deepseek_v4",
                        "cohere2_moe")},
    # DeepSeek-V2's HF port ignores norm_topk_prob (always scales, never renormalises); Jamba softmax top-k;
    # LongCat (softmax + selection bias, no renorm); Llama-4: sigmoid of the top-1 logit, no division
    **{t: "P" for t in ("deepseek_v2", "jamba", "longcat_flash", "llama4", "llama4_text")},
    # PhiMoE: SparseMixer at inference (a top-2 with a threshold mask, weights not summing to 1): its own kind
    "phimoe": "sparsemixer"}
# round 3, batch 3a (renorm_by_code_unlisted): renormalisation fixed in code with no key p5 reads. R: CENO
# (modeling_ceno.py:1184-1192, p / sum or a softmax over the k), AliceAI (modeling_alice_ai.py:552-554), ZAQERIS
# (modeling_zaqeris_axiom.py:351-352), Gemma3-MoE (modeling_gemma3_moe.py:58-60), IQuest-Q1 (vllm model.py:125, 157
# renormalize=True), Mistral-Large-4 (modular_mistral_large4.py:528 `del self.norm_topk_prob`, :549-552 always / sum).
# P: BananaMind-2 (modeling_bananamind2moe.py:185-198, gate = the max softmax probability), Step-3 (modeling_step3.py:
# 365-368; norm_expert_weight is not read there), MolmoE (modeling_molmoe.py:829-833). The k-dependent families (Bailing
# V2 / V2.5 / V3 by architecture, Arctic, Meditor, LLaMA-MoE) are RENORM_BY_ARCH / _renorm_by_code below.
# round 3, batch 3a (router_score_unread, companion): Sarvam-105B (modeling_sarvam_moe.py:289 norm_topk_prob = True
# hard-coded), Grug-MoE (skyrl_train_models_grug_moe.py:544-553 sigmoid * 2.5 / sum), Muse-Spark (modular_muse_spark.py:
# 335 / (sum + 1e-15)), Mark-SLM-V3 (model_type deepseek2: modeling_mark_v3.py:195-200 / sum.clamp(1e-6)) -- always R
RENORM_FIXED = _R3Dict(RENORM_FIXED, add={
    "renorm_by_code_unlisted": {**{t: "R" for t in ("ceno", "alice_ai", "zaqeris_axiom", "gemma3_moe", "iquest_q1",
                                                    "mistral_large4", "mistral_large4_text")},
                                **{t: "P" for t in ("bananamind2_moe", "step3_text", "step3_vl", "molmo")}},
    "router_score_unread": {t: "R" for t in ("sarvam_mla", "grug_moe", "muse_spark", "muse_spark_text", "deepseek2")}})
# round 3, batch 3a (renorm_by_code_unlisted): the BailingMoeV2 / V2.5 / V3 gates and their forks divide by the selected
# mass iff top_k > 1 and never read norm_topk_prob (Ling-1T modeling_bailing_moe_v2.py:345, Ling-2.6 _v2_5.py:362,
# Ling-3.1 / Edge0 _v3.py:409 / :472, Param2 modeling_param2moe.py:337, LLaDA2.1 / 2.2 modeling_llada2_moe.py:363-367 /
# :291-293, Ming-3 modeling_ming_v3_asr.py:487-488); bailing_moe v1 (BailingMoeForCausalLM, Ling-lite :451-453) does read
# it, so this is by ARCHITECTURE (Edge0 has no model_type)
RENORM_BY_ARCH = _R3Set(add={"renorm_by_code_unlisted": {
    "BailingMoeV2ForCausalLM", "BailingMoeV2_5ForCausalLM", "BailingMoeV3ForCausalLM", "LLaDA2MoeModelLM",
    "Param2MoEForCausalLM", "MingV3ASRForConditionalGeneration"}})


def _renorm_by_code(d, outer, mt, k):
    """round 3, batch 3a (renorm_by_code_unlisted): (renorm, renorm_cap) a family's code fixes from its top-k (no key
    read), or None. Arctic (modeling_arctic.py:973-976): / sum iff top_k > 1; Meditor (modeling_meditor.py:231-234):
    p / (p.detach() + 1e-6), forward value 1 -- a top-1 renormalised weight at k = 1 (the 1e-6 a limit well inside fp32),
    a constant-1 weight at k > 1 (no R / P kind: family-only); LLaMA-MoE (modeling_llama_moe_hf.py:494-497): a softmax
    over the k selected logits when gate_use_softmax (default True, :786), else family-only."""
    if not _r3("renorm_by_code_unlisted"):
        return None
    archs = [str(a) for a in (d.get("architectures") or (outer or {}).get("architectures") or [])]
    if any(a in RENORM_BY_ARCH for a in archs):
        r = "R" if k > 1 else "P"
    elif mt == "arctic":
        r = "R" if k > 1 else "P"
    elif mt == "meditor":
        r = "R" if k <= 1 else "?meditor"
    elif mt == "llama_moe":
        r = "R" if d.get("gate_use_softmax", True) is not False else "?llama_moe"
    else:
        return None
    return r, (r,)
# the renorm read from a config key the code honours (norm_topk_prob unless listed): Qwen1.5 / 2 / 3-MoE, Qwen3-Next /
# Omni, OLMoE, FlexOlmo, DeepSeek-V3 / V3.2 / GLM-4.5 / 4.7 / 5 / dots1 / A.X / EXAONE-MoE / Solar-Open / Mistral-4 /
# Kimi-Linear / Nemotron-H / LFM2-MoE / Mellum / MiMo-V2 / Hy4 / Doge (T5), the DeepSeek-V3-architecture remote code
# (Kimi-K2, JoyAI), Flint, Step-3 / 3.5 (norm_expert_weight)
RENORM_KEYS = ("norm_topk_prob", "moe_renormalize", "norm_expert_weight")
# round 3, batch 3a (renorm_by_code_unlisted): Motif-3 renormalises iff `route_norm` (modeling_motif.py:806, :867-870),
# read only for that family (the key is no renorm switch elsewhere): renorm_cap P | R, a host config lever as
# norm_topk_prob is for the RENORM_KEYS families
RENORM_KEYS_BY_TYPE = _R3Dict(add={"renorm_by_code_unlisted": {"motif": ("route_norm",)}})
# the router scores a family's code computes from a config key (others: the one scoring fixed in code):
# DeepSeek-V4 ACT2FN[scoring_func] (sigmoid / sqrtsoftplus / ...), Cohere2-MoE and Step-3.5 (softmax or sigmoid)
SCORE_CONFIG_TYPES = {**{t: ("sigmoid", "sqrtsoftplus") for t in ("deepseek_v4", "deepseek_v41_text", "memt_deepseek_v4")},
                      "cohere2_moe": ("sigmoid", "softmax"), "step3p5": ("sigmoid", "softmax")}
# round 3, batch 3a (router_score_unread): the score read from a key the code honours -- K2-Horizon router_score_func
# (modeling_k2_horizon.py:538, 562-566), Motif-3 score_func (modeling_motif.py:846-851), Kohaku scoring_func softmax /
# sqrtsoftplus / sigmoid (modeling_kohaku.py:147-152): a host of these families sets it (host config lever)
SCORE_CONFIG_TYPES = _R3Dict(SCORE_CONFIG_TYPES, add={"router_score_unread": {
    "k2_horizon": ("sigmoid", "softmax"), "motif": ("sigmoid", "softmax"),
    "kohaku": ("sigmoid", "softmax", "sqrtsoftplus")}})
# group-limited routing code (n_group / topk_group are config values) and its GROUP SCORE: max (DeepSeek-V2) or the
# sum of the top 2 (DeepSeek-V3 and every port of its router)
GROUP_SCORE_TYPES = {
    **{t: "max" for t in ("deepseek_v2",)},
    **{t: "top2sum" for t in ("deepseek_v3", "deepseek_v32", "kimi_k2", "joyai_llm_flash", "glm4_moe", "glm4_moe_lite",
                              "glm_moe_dsa", "glm5_next", "glm5_next_text", "dots1", "kimi_linear", "exaone_moe",
                              "solar_open", "solar_open2", "mistral4", "mistral_large4_text", "mimo_v2",
                              "mimo_v2_flash", "nemotron_h", "nemotron_h_puzzle", "axk1", "axk2", "hy_v4",
                              "bailing_moe", "bailing_hybrid", "llada2_moe", "xing4_0", "naive_n05_flash")}}
# round 3, batch 3a. router_score_unread: Beam's BeamTopKRouter = DeepseekV3TopkRouter (modeling_beam.py:339-375, -inf
# fill :369). group_mask_zero_fill: "top2sum_zero" = the top-2-sum group limit whose code fills the excluded experts'
# BIASED scores with 0.0, not -inf (tf5 modular_axk2.py:236-237 "fill with 0.0 intentionally"; the first-release remote
# modeling_deepseek.py of Kimi-K2 / Moonlight / MAI-DS-R1 / JoyAI :459): with a negative selection bias an expert of a
# dropped group can still win (round3_inputs G1), so it is a group code of its own (the per-config remote readings:
# _group_cap). router_group_balanced: Intern-S1-Pro's grouped router (modeling_interns1_pro.py:361-388) takes top-(k / G)
# inside EVERY group, no group score and no group dropped ("per_group"; active iff router_n_groups > 1, :70-73)
GROUP_SCORE_TYPES = _R3Dict(GROUP_SCORE_TYPES, add={
    "router_score_unread": {"beam": "top2sum"},
    "group_mask_zero_fill": {"axk2": "top2sum_zero"},
    "router_group_balanced": {"interns1_pro": "per_group", "interns1_pro_text": "per_group"}})


# fr-moe (group_code_by_arch): the BailingMoeV3 gate's group_limited_topk read in code -- Edge0-8B-A1B (typeless config,
# architectures BailingMoeV3ForCausalLM; modeling_bailing_moe_v3.py:437-457) and Ming-3.0-ASR (model_type ming_v3_asr;
# modeling_ming_v3_asr.py:447-469): scores.view(T, n_group, -1).topk(2).sum() per group, topk_group kept, the excluded
# experts filled -inf -- the top-2-sum code of GROUP_SCORE_TYPES, so the group cap was an unread "?"
GROUP_SCORE_TYPES = _R3Dict(GROUP_SCORE_TYPES, add={"group_code_by_arch": {"ming_v3_asr": "top2sum"}})
GROUP_SCORE_BY_ARCH = _R3Dict(add={"group_code_by_arch": {"BailingMoeV3ForCausalLM": "top2sum"}})


def _group_cap(d, outer, mt):
    """The group-score code of a MoE config (GROUP_SCORE_TYPES), with round 3's per-repo readings
    (group_mask_zero_fill): JoyAI-LLM-Flash and the first-release DeepSeek-V3 / Kimi-K2 remote modeling_deepseek.py
    (the one whose MoEGate still reads config.seq_aux: MAI-DS-R1 :401, Kimi-K2 :401, K2.5 :421; the -inf revision has
    no seq_aux) fill 0.0. Config proxy: a remote modeling_deepseek / modeling_kimi_k25 + a seq_aux key + noaux_tc
    (the spec's known mismatch: ByteDance academic-ds-9B reads seq_aux but fills -inf -- the conservative side)."""
    g = GROUP_SCORE_TYPES.get(mt)
    if g is None and not mt:
        archs = [str(a) for a in (d.get("architectures") or (outer or {}).get("architectures") or [])]
        g = next((GROUP_SCORE_BY_ARCH[a] for a in archs if a in GROUP_SCORE_BY_ARCH), None)
    if _r3("group_mask_zero_fill") and g == "top2sum":
        am = _auto_map_str(d, outer)
        remote_ds = "modeling_deepseek" in am or "modeling_kimi_k25" in am
        if (mt == "joyai_llm_flash" and remote_ds) or (
                mt in ("deepseek_v3", "kimi_k2") and remote_ds and "seq_aux" in d
                and str(d.get("topk_method") or "").lower() == "noaux_tc"):
            g = "top2sum_zero"
    return g


# a router with a LOGIT bias (nn.Linear(bias=True): gpt-oss GptOssTopKRouter): it shifts the weights too (so it is no
# selection-only bias), but under renormalisation a -M bias still sinks an extra expert below the copies
ROUTER_LOGIT_BIAS_TYPES = {"gpt_oss", "gpt_oss_puzzle"}


def _router_flags(d):
    """(sigmoid, sqrtsoftplus, selection bias) of a MoE config's router: the scoring keys, the code-fixed
    SIGMOID_ROUTER_TYPES / ROUTER_BIAS_TYPES, the aux-loss-free bias keys."""
    mt = str(d.get("model_type") or "").lower()
    sc = str(d.get("scoring_func") or d.get("score_func") or d.get("score_function") or d.get("router_score_func")
             or d.get("router_score_function") or d.get("moe_router_activation_func")
             or d.get("moe_router_activation") or "").lower()          # Step-3.5: moe_router_activation
    sig = "sigmoid" in sc or bool(d.get("moe_router_use_sigmoid")) or mt in SIGMOID_ROUTER_TYPES
    ssp = "sqrtsoftplus" in sc
    bias = bool(d.get("topk_method") == "noaux_tc" or d.get("moe_router_enable_expert_bias") or d.get("use_expert_bias")
                or d.get("use_routing_bias") or d.get("moe_gate_bias") or d.get("use_moe_router_bias")
                or mt in SIGMOID_ROUTER_TYPES and mt not in ROUTER_NO_BIAS_TYPES or mt in ROUTER_BIAS_TYPES)
    if _r3("router_bias_unread") and not bias:
        # round 3, batch 3a: bias switches p5 did not read -- AliceAI e_score_correction_bias iff router_bias_correction
        # (modeling_alice_ai.py:535-540), Motif-3 expert_bias iff load_balance_coeff is not None (modeling_motif.py:
        # 978-983), Youtu-A7B's moe_router_enable_expert_bias with its class default True (configuration_youtu_vl_moe.py:
        # 78-80; the snapshot omits the key)
        bias = bool(d.get("router_bias_correction") or (mt == "motif" and d.get("load_balance_coeff") is not None)
                    or (mt == "youtu_vl_moe" and _cfg_value(d, "moe_router_enable_expert_bias")))
    return sig, ssp, bias


# round 3, batch 3a (router_bias_unread): a LOGIT-domain selection bias -- top-k on l + b, weights from the unbiased
# score sigma(l) (Grug-MoE skyrl_train_models_grug_moe.py:544-553, Kolibri-1 modular_kolibri1.py:136-137; Youtu-A7B's
# quantile balancing topk(logits - qb_beta) modeling_youtu_vl_moe.py:518-524). For a sigmoid router top-k on sigma(l) + b
# and on l + b' order experts differently unless b, b' are constant: a domain of its own (FEATURES router_bias_logit)
ROUTER_BIAS_LOGIT_TYPES = _R3Set(add={"router_bias_unread": {"grug_moe", "kolibri1"}})
_CLASS_DEFAULT_FALLBACK["moe_router_enable_expert_bias"] = _R3Dict({}, add={"router_bias_unread": {"youtu_vl_moe": True}})


def _router_bias_logit(d):
    """The selection bias is in the logit domain (ROUTER_BIAS_LOGIT_TYPES, Youtu's quantile balancing)."""
    mt = str(d.get("model_type") or "").lower()
    if mt in ROUTER_BIAS_LOGIT_TYPES:
        return True
    return bool(_r3("router_bias_unread") and mt == "youtu_vl_moe"
                and (str(d.get("moe_router_expert_bias_update_method") or "").lower() == "quantile"
                     or str(d.get("topk_method") or "").lower() in ("quantile_balancing", "quantile")))


def _hier_moe(d):
    """round 3, batch 3a (hierarchical_moe_unread): Meditor-2's two-level hard top-1 -- (macro domains, sub-experts per
    domain, sub-expert width) (configuration_meditor2.py:28-31, 59-63 defaults 3 / 5 / 768), or None."""
    if not (_r3("hierarchical_moe_unread") and str(d.get("model_type") or "").lower() == "meditor2"):
        return None
    D = _num(d.get("num_macro_domains", 3))
    S = _num(d.get("num_sub_experts_per_domain", 5))
    return (D, S, _num(d.get("sub_intermediate_size", 768))) if D * S > 1 else None


def _zero_experts(d):
    """round 3, batch 3a (identity_experts): LongCat's zero-computation router slots whose expert is the identity
    (zero_expert_num with zero_expert_type identity: tf5 modeling_longcat_flash.py:143-218), else 0."""
    if not _r3("identity_experts"):
        return 0
    z = _num(d.get("zero_expert_num"))
    return int(z) if z > 0 and str(d.get("zero_expert_type") or "identity").lower() == "identity" else 0


def _block_routing(d):
    """round 3, batch 3a (block_routing): LLaDA2.2's block-routing gate (modeling_llada2_moe.py:228-275): per block of
    block_size tokens only the expert_capacity best-scoring experts (block max of sigmoid + bias) are allowed, then a
    per-token top-k among them -- (block_size, expert_capacity), or None. LLaDA2.0 / 2.1 (no such keys) run the group-
    limited gate."""
    if not (_r3("block_routing") and str(d.get("model_type") or "").lower() == "llada2_moe"):
        return None
    E = _num(d.get("num_experts") or d.get("n_routed_experts"))
    cap, bs = _num(d.get("expert_capacity")), _num(d.get("block_size"))
    return (int(bs), int(cap)) if cap and bs and cap < E else None


def _latent_norm(d, mt):
    """round 3, batch 3a (kimi_latent_moe_norm): (expert latent width, the latent norm's site) of a latent MoE with an
    RMSNorm in the latent: "sum" = on the weighted routed sum before the up-projection (ERNIE-5.5 latent_norm, built
    unconditionally: vLLM-ERNIE moe.py:180, 91-101), "sum?" = the same with a norm on / off flag a host sets
    (Kimi-K3 latent_moe_use_norm: modeling_kimi_linear.py:804-813), "in" = on the latent expert INPUT (Muse-Spark
    pre_expert_norm, modular_muse_spark.py:376-394); else (latent, "")."""
    lat = _num(d.get("moe_latent_size"))
    if not _r3("kimi_latent_moe_norm"):
        return lat, ""
    if mt == "kimi_linear" and _num(d.get("routed_expert_hidden_size")):
        lat = _num(d["routed_expert_hidden_size"])
        return lat, ("sum?" if d.get("latent_moe_use_norm") is True else "")
    if mt == "ernie5_5_moe" and lat:
        return lat, ("sum" if d.get("latent_moe_use_norm") is not False else "")
    if mt in ("muse_spark", "muse_spark_text"):
        return (_num(d.get("moe_hidden_size")) or 4096), "in"
    return lat, ""


def _moe_sig(d, n_exp, k, ew, shared, dense_w, hidden, n_layers, eff_layers, outer=None):
    """The routing facts the MoE rules compare (moe_fits), as a hashable tuple MOE_SIG fields; dense models
    get E = 0 and every FFN layer dense."""
    mt = str(d.get("model_type") or "").lower()
    sub = _ffn_sublayers(d)
    scale = (lambda c: max(0, round(c * eff_layers / n_layers))) if n_layers and eff_layers != n_layers else (lambda c: c)
    if not n_exp:
        nd = sub[1] if sub else eff_layers
        return (0, 0, 0, 0, "", (), "", (), 0, 0, "", 0, nd, dense_w, 0, 0) + MOE_SIG_R3_DENSE
    if sub:
        n_moe, n_dense = sub
    else:
        mask = _moe_layer_mask(d, n_layers)
        n_moe = n_layers if mask is None else sum(mask)
        n_dense = 0 if mask is None else len(mask) - sum(mask)
        n_moe, n_dense = scale(n_moe), scale(n_dense)
        if n_moe == 0:                                  # a mask that names no MoE layer: the counts are a template
            n_moe, n_dense = eff_layers, 0
    sig, ssp, bias = _router_flags(d)
    score = "sqrtsoftplus" if ssp else "sparsemixer" if mt == "phimoe" else "sigmoid" if sig else "softmax"
    if _r3("dual_sigmoid_router_sum") and mt == "ernie5_5_moe" and d.get("moe_split_feature_routing"):
        # round 3, batch 3a: ERNIE-5.5's split-feature routing, gate = Linear(h, 2E), score sigmoid(l[:E]) +
        # sigmoid(l[E:]) (vLLM-ERNIE moe.py:151-157, :217): no single sigmoid router equals the sum. Its own score; the
        # host side (tie the two rows: 2 sigma(a), selection with a doubled bias) needs theorem sigmoid2_tied_rows, so
        # an ERNIE-5.5 host plays only sigmoid2 targets (score_cap = its own score)
        score = "sigmoid2"
    hier = _hier_moe(d)
    if hier:
        # round 3, batch 3a: Meditor-2's argmax domain, then argmax sub-expert, weight the constant routed_scale
        # (modeling_meditor2.py:232-297): no flat top-1 router induces the two-level partition (needs theorem
        # hier_argmax_not_flat); the host side (zero macro rows, torch.argmax's first index) needs theorem
        # argmax_tie_first_select -- so the score names the shape and is its own cap
        score = f"hier{hier[0]}x{hier[1]}"
    score_cap = tuple(sorted(set(SCORE_CONFIG_TYPES.get(mt, ())) | {score})) if score in ("softmax", "sigmoid",
                                                                                        "sqrtsoftplus") else (score,)
    if _r3("score_not_free"):
        # round 3 (transitivity on the real shapes; a standing design rule): the router SCORE FUNCTION (softmax /
        # sigmoid / sqrtsoftplus) is a feature switch, not a free shape-preserving value, so a host plays only targets
        # of its own score function. As a host lever (pass 5's SCORE_CONFIG_TYPES, round 3's router_score_unread) it was
        # not class-closed: small softmax MoE targets <= K2-Horizon-375B (scoring_func set to softmax) <= Ling-1T
        # (sigmoid only) held without the composite (9 payable / mixgap violations); a composed edge is not a
        # construction there (a softmax target on a sigmoid-only host: router_sigmoid). Cost on the snapshot: 23 edges
        score_cap = (score,)
    rbc =_renorm_by_code(d, outer, mt, int(k or 1))
    if hier:
        renorm, renorm_cap = "R", ("R",)                 # a top-1 constant weight: renormalised (scale folded)
    elif rbc:
        renorm, renorm_cap = rbc
    elif mt in RENORM_FIXED:
        renorm = RENORM_FIXED[mt]
        renorm_cap = (renorm,)
    else:
        keys = RENORM_KEYS + tuple(RENORM_KEYS_BY_TYPE.get(mt, ()))
        key = next((kk for kk in keys if d.get(kk) is not None), None)
        v = d.get(key) if key else _cfg_value(d, "norm_topk_prob")
        if v is None:
            renorm, renorm_cap = "?" + mt, ("?" + mt,)       # not read: only the same family plays it
        else:
            renorm, renorm_cap = ("R" if v else "P"), ("P", "R")
    G = _num(d.get("n_group")) or _num(d.get("num_expert_group"))
    g = _num(d.get("topk_group"))
    active = G > 1 and 0 < g < G
    gcap = _group_cap(d, outer, mt) or ("?" + mt if active else "")
    if _r3("router_group_balanced") and gcap == "per_group":
        # round 3, batch 3a: router_n_groups > 1 builds the grouped router (modeling_interns1_pro.py:70-73): G groups,
        # top-(k / G) in each -- every group "kept" (topk_group = G); router_n_groups = 1 falls back to the plain top-k
        # router, so a host plays ungrouped targets (host config lever: router_n_groups, a routing number)
        R = _num(d.get("router_n_groups"))
        G = g = R
        active = R > 1
    blk = _block_routing(d)
    if blk:
        # round 3, batch 3a: the block-routing gate never reads n_group / topk_group (modeling_llada2_moe.py:228-236)
        G = g = 0
        active, gcap = False, ""
    lat, lnorm = _latent_norm(d, mt)
    branch = int(_r3("gemma4_branch_post_norms") and _dense_mlp_beside_moe(d))
    par_res = (_num(d.get("hidden_size")) if _r3("arctic_dense_residual_parallel_moe") and mt == "arctic"
               and d.get("parallel_attn_mlp_res") is True else 0)
    return (int(n_exp), int(k or 1), int(ew), int(shared), renorm, renorm_cap, score, score_cap,
            int(G) if active else 0, int(g) if active else 0, gcap, int(n_moe), int(n_dense),
            int(dense_w) if n_dense else 0,                 # (no dense layer: the dense width is never read)
            # fr-moe (moe_null_experts): 1 = a SELECTION-ONLY bias (score or logit domain: the weights read the unbiased
            # score), 2 = gpt-oss's Linear logit bias only (it shifts the weights too: it sinks extras under renorm but
            # cannot select an expert at zero weight); before fr-moe both were 1
            int(lat), int(1 if bias else (2 if _r3("moe_null_experts") else 1) if mt in ROUTER_LOGIT_BIAS_TYPES else 0),
            # round 3, batch 3a: the MoE-block forms (MOE_SIG_R3)
            branch, lnorm, _zero_experts(d), blk[0] if blk else 0, blk[1] if blk else 0, int(par_res))


MOE_SIG = ("E", "k", "ew", "shared", "renorm", "renorm_cap", "score", "score_cap", "n_group", "topk_group",
           "group_cap", "n_moe", "n_dense", "dense_w", "latent", "bias")
# round 3, batch 3a: MoE-block forms the routing rules compare (appended, so the positional reads of the first 16 stay):
#   branch       Gemma-4 / DiffusionGemma: the dense MLP and the experts each post-normed, the sum normed again
#                (gemma4_branch_post_norms; modeling_gemma4.py:1376-1382, :1410-1429)
#   latent_norm  "" / "sum" / "sum?" (a host-settable flag) / "in": an RMSNorm in the expert latent (kimi_latent_moe_norm)
#   ident        LongCat's identity router slots (identity_experts)
#   blk, blk_cap LLaDA2.2's block routing: block_size, expert_capacity (block_routing)
#   par_res      Arctic: the MoE in parallel with attention plus a dense residual MLP of this width on the post-attention
#                stream (arctic_dense_residual_parallel_moe; modeling_arctic.py:1042-1050, :1083-1110)
MOE_SIG_R3 = ("branch", "latent_norm", "ident", "blk", "blk_cap", "par_res")
MOE_SIG_R3_DENSE = (0, "", 0, 0, 0, 0)
MOE_SIG = MOE_SIG + MOE_SIG_R3


def _moe_fields(sig):
    """dict(zip(MOE_SIG, sig)) with the round-3 fields' defaults for a shorter (pre-round-3, synthetic) tuple."""
    T = dict(zip(MOE_SIG_R3, MOE_SIG_R3_DENSE))
    T.update(zip(MOE_SIG, sig))
    return T


# round 3, batch 3a: the latent-norm sites a host's latent norm plays (kimi_latent_moe_norm). A "sum" target (the norm on
# the routed sum) needs a "sum" host (the same form; a wider latent pads with the host eps: qknorm_pad, eps free); a
# "sum?" host (Kimi-K3: latent_moe_use_norm, a norm on / off flag -- host config lever) plays "sum" targets with the
# norm on and norm-free targets with it off (then a linear latent: the rank rule); as a TARGET a "sum?" model is its own
# config, so it stands for its class and needs a "sum?" host (transitivity). "in" (Muse-Spark: the norm on the expert
# input) only an "in" host. A "sum" / "in" host with no flag playing a norm-free target is the eps lever (limit; not
# plumbed here: a wall -- target side needs theorem latent_sum_norm_not_plain)
LATENT_NORM_PLAYS = {"": ("", "sum?"), "sum": ("sum", "sum?"), "sum?": ("sum?",), "in": ("in",)}


def _r3_moe_forms_dense_host(T):
    """round 3, batch 3a: a MoE block form a dense host's select / mask construction (one FFN unit per expert, a unit
    gate) does not cover, or None."""
    if T["score"].startswith("hier"):
        return "a two-level argmax router (hierarchical_moe_unread)"
    if T["ident"]:
        return "identity expert slots (identity_experts)"
    if T["blk"]:
        return "block routing (block_routing)"
    if T["latent_norm"]:
        return "a latent-norm MoE (kimi_latent_moe_norm)"
    if T["branch"]:
        return "branch post-norms (gemma4_branch_post_norms)"
    if T["par_res"]:
        return "a parallel MoE with a residual MLP (arctic_dense_residual_parallel_moe)"
    return None


def moe_why(t, h, strict=False):
    """None if host h = (moe_sig, hidden, ffn_total) plays target t's FFN routing, else the reason (the MoE rules of
    emul_matrix; Lean MoERouting.lean, moe/dense_host_prenorm.py on branch moe-loop-rescue)."""
    (st, ht, ft), (sh, hh, fh) = t, h
    if not st or not sh:
        return None                                     # a synthetic point without routing facts
    T, H = _moe_fields(st), _moe_fields(sh)
    Et, Eh = T["E"], H["E"]
    if not Et and not Eh:
        return None
    if Et and not Eh:                                   # dense host for a MoE target (the pre-norm select / mask)
        if not (T["k"] == 1 and T["renorm"] == "R"):
            return f"MoE<-dense: top-{T['k']} renorm {T['renorm']} is not a top-1 unit gate"
        form = _r3_moe_forms_dense_host(T)
        if form:
            return f"MoE<-dense: {form}"
        if not _cr("moe_dense"):
            return None
        if hh < ht + Et:
            return f"MoE<-dense: hidden {hh} < {ht} + {Et} select channels"
        if fh < ft + 2 * Et * (Et - 1):
            return f"MoE<-dense: ffn {fh} < {ft} + {2 * Et * (Et - 1)} erase units"
        return None
    # round 3, batch 3a (gemma4_branch_post_norms): a branch-normed target needs a branch-normed host (the dense branch on
    # the dense branch, the experts on the experts); a branch-normed host playing anything else runs one branch through
    # N_post(N_post1(.)) -- the fold N(w sqrt(1 + eps), N(1, y / c)) = N(w, y), c^2 = (1 + eps) / eps, needs theorem
    # double_rmsnorm_fold (PLAN, "Theorems the specs ask for"): a wall until then
    if T["branch"] != H["branch"]:
        return ("branch post-norms: the target's branch-normed sum needs a branch-normed host" if T["branch"] else
                "branch post-norms: a host branch is double-normed (needs theorem double_rmsnorm_fold)")
    # round 3, batch 3a (arctic_dense_residual_parallel_moe): an Arctic host's MoE reads the LAYER input, beside
    # attention; a sequential target's MoE layer needs the 2x-depth pairing (seq_sub_parallel), not plumbed here: a wall
    # for MoE targets without the same block. Dense target layers run on its residual MLP (MoE zeroed: exact at 1x,
    # zeroed_sublayer), width par_res -- the routed / shared experts cannot stand in for a dense FFN there
    if H["par_res"] and Et and not T["par_res"]:
        return "a parallel MoE + residual-MLP host plays a sequential MoE layer only at 2x depth (not plumbed)"
    # a host MoE layer as a dense layer: zero router, the units dealt over the k selected experts (x k), + shared
    rank_ok = (not H["latent"] or H["latent"] >= ht) and H["latent_norm"] in ("", "sum?")
    capd = (H["k"] * H["ew"] if rank_ok else 0) + H["shared"]
    if H["branch"]:
        capd = max(H["k"] * H["ew"] if rank_ok else 0, H["shared"])     # one branch at a time, not their sum
    if H["par_res"]:
        capd = H["par_res"]
    dense_slots = lambda spare: ((H["n_dense"] if H["dense_w"] >= T["dense_w"] else 0)
                                 + (spare if capd >= T["dense_w"] else 0))
    if not Et:
        if T["n_dense"] > dense_slots(H["n_moe"]):
            return f"layers: {T['n_dense']} dense target layers, host dense / MoE-as-dense layers too few or narrow"
        return None
    first = None
    for m, r in _moe_copy_plans(T, H, ht, hh, strict):
        w = _moe_why_plan(T, H, ht, hh, m, r, dense_slots, strict)
        if w is None:
            return None
        first = first or w
    return first


def _moe_copy_plans(T, H, ht, hh, strict=False):
    """The (m, r) plans moe_why tries, in order: m router-row copies of each target expert and r NULL experts filling a
    top-k remainder k_h = m k_t + r (fr-moe, moe_null_experts). The duplication plan (m = k_h / k_t, r = 0) first when
    k_t | k_h, else a plan that fails on the multiple; then, on a host with a SELECTION-ONLY bias and a free hidden dim,
    every m <= k_h / k_t with r = k_h - m k_t > 0, largest m first. The nulls only on a host whose OWN score is the
    target's (a transitivity guard, as the unrenormalised one: a host SET to the target's score, K2-Horizon's
    router_score_func, would extend the snapshot's K2-Horizon triples -- softmax target <- K2-Horizon <- Ling-1T)."""
    kt, kh = T["k"], H["k"]
    # (fr-cost COST_RELAX moe_topk, a counterfactual only: the multiple not charged)
    plans = [(max(kh // kt, 1), 0)] if (kh % kt == 0 or not _cr("moe_topk")) else [(0, 0)]
    if (_r3("moe_null_experts") and not strict and H["bias"] == 1 and hh > ht
            and T["score"] == H["score"]):
        plans += [(m, kh - m * kt) for m in range(kh // kt, 0, -1) if kh - m * kt > 0]
    return plans


def _moe_why_plan(T, H, ht, hh, m, r, dense_slots, strict=False):
    """moe_why's routing checks for one copy plan: m copies of each target expert (Lean moeR_dup_renorm / _plain), and
    r null experts (fr-moe, moe_null_experts): host experts with zero weights, a selection bias above every copy's
    (always selected) and a router row reading -M off the constant channel of a free hidden dim (const_norm_carry),
    so their score sigma(-M c / R) (renormalised) or exp(-M c / R) (softmax over all) is 0 in fp32 and every copy keeps
    the target's gate (Lean MoENullExperts.lean moeR_null_topk / moeR_null_renorm_bound; unrenormalised:
    moeR_dup_plain_extras_bound): exact in the limit M -> inf, reached at the fp32 underflow. The free hidden dim and the bias are _moe_copy_plans' conditions."""
    Et, Eh = T["E"], H["E"]
    mc = _cr("moe_cap")                                # (fr-cost COST_RELAX moe_cap: a counterfactual only)
    if not m:
        return f"top-k {H['k']} not a multiple of {T['k']}"
    if r and (T["n_group"] or T["ident"] or H["ident"] or T["blk"] or T["score"] == "sparsemixer"):
        # (null experts beside group-limited routing, identity slots, block routing or SparseMixer: not built)
        return f"top-k {H['k']} not a multiple of {T['k']}"
    if Eh < m * Et + r and mc:
        return f"experts {Eh} < {m} x {Et}" + (f" + {r} null" if r else "")
    if m * H["ew"] < T["ew"] and mc:
        return f"expert width {m} x {H['ew']} < {T['ew']}"
    if H["shared"] < T["shared"] and mc:
        return f"shared expert {H['shared']} < {T['shared']}"
    if (H["latent"] or hh) < (T["latent"] or ht):
        return f"expert latent {H['latent'] or hh} < {T['latent'] or ht}"
    if H["latent_norm"] not in LATENT_NORM_PLAYS.get(T["latent_norm"], (T["latent_norm"],)):
        return f"latent norm {T['latent_norm'] or 'none'}<-{H['latent_norm'] or 'none'}"
    if H["par_res"] < T["par_res"]:
        return f"residual MLP {H['par_res']} < {T['par_res']}"
    if H["n_moe"] < T["n_moe"] and mc:
        return f"layers: host MoE layers {H['n_moe']} < {T['n_moe']}"
    if T["n_dense"] > dense_slots(H["n_moe"] - T["n_moe"]):
        return f"layers: {T['n_dense']} dense target layers, host dense / spare MoE layers too few or narrow"
    if T["score"] not in H["score_cap"]:
        # (fr-moe, refuted: a renormalised softmax target on a renormalised sigmoid host through the exp regime -- every
        # copy row reads -rho off a free dim's constant channel, sigma(z - s) ~ e^(z - s) -- holds in fp64 on random
        # weights (moe/router_frontier.py ref_exp, rel 2.7e-7) but the shift s = rho c / R falls with the token's RMS,
        # and a massive-activation first token (rms ~100x) sits in the sigmoid regime: granite-3.0 / 3.1-1b-a400m on a
        # Glm4Moe host 99.28% / 99.54% top-1, 79% / 88% identical 32-token continuations -- below the standard; no rule)
        return f"router score {T['score']}<-{'/'.join(H['score_cap'])}"
    if T["renorm"] not in H["renorm_cap"]:
        return f"renorm {T['renorm']}<-{'/'.join(H['renorm_cap'])}"
    if T["score"] == "sparsemixer" and m != 1 and _r3("spot_phimoe_dup"):
        # round 3, batch 3a (spot check, verify/spot/SPOT.md): SparseMixer is a top-2 in code whatever
        # num_experts_per_tok says, and its second pick masks out only the first expert's INDEX, so a copy of the
        # top-1 expert wins the second slot (catalog fm_phimoe_dup2: top-1 43.8%, 0/8 identical): copies never work
        return f"SparseMixer with {m} copies (its second pick masks one index)"
    if (T["ident"] or H["ident"]) and H["ident"] != m * T["ident"]:
        # round 3, batch 3a (identity_experts): an identity slot outputs w x, which no bias-free SwiGLU expert forms
        # (needs theorem swiglu_not_identity), and it takes softmax mass in the unrenormalised denominator, so a host
        # without the target's slots (or with extra ones) is never exact (moeR_dup_plain_extras_ne); m copies of each
        # slot with its router row split its probability (moeR_copies_topk)
        return f"identity slots {H['ident']} != {m} x {T['ident']}"
    if T["blk"] and not (H["blk"] == T["blk"] and H["blk_cap"] == T["blk_cap"] and m == 1 and Eh == Et):
        # round 3, batch 3a (block_routing): a token's experts depend on its block (block_routing_not_tokenwise: no
        # per-token host); copies would tie at block level (needs theorem block_routing_copies), so only the same
        # block routing at m = 1 with the target's experts. A block-routed HOST plays per-token targets with
        # expert_capacity = num_experts (host config lever: every expert allowed, topk_all)
        return f"block routing {T['blk']}/{T['blk_cap']}<-{H['blk'] or 'none'}/{H['blk_cap']} (m {m}, E {Eh}/{Et})"
    if T["renorm"] == "P" and T["score"] == "softmax":
        # exactly m x the experts (moeR_dup_plain); and the host's OWN router unrenormalised softmax: forced by
        # transitivity (tiny_qwen3_moe (P) <- Qwen3-Coder-16B-KREAP (R, settable to P) <- Qwen3-235B (R) held, yet
        # the 235B's extra experts break the P target: the middle host is played only as its own R config).
        # fr-moe (moe_plain_extras_sink): extras (and null experts) beyond the m E_t copies are no dilution when they
        # read -M off a free hidden dim's constant channel: exp(-M c / R) / Z_h is 0 in fp32 (Lean
        # moeR_dup_plain_extras_bound: relative error Z_D / (m Z), a limit reached at the fp32 underflow)
        if not (H["renorm"] == "P" and H["score"] == "softmax"
                and (Eh == m * Et + r or (_r3("moe_plain_extras_sink") and not strict and hh > ht))):
            return "unrenormalised softmax: needs a host whose own router is too, with exactly m x the experts"
    elif Eh > m * Et + r and not (H["bias"] or hh > ht):
        return "extra experts need a selection bias or a free hidden dim to score below the copies"
    if T["n_group"] and T["group_cap"] != H["group_cap"]:
        return f"group routing {T['group_cap']}<-{H['group_cap'] or 'none'}"
    if T["n_group"] and str(T["group_cap"]).startswith("top2sum") and m != 1:
        return f"top-2-sum group score with {m} copies"
    if T["n_group"] and T["group_cap"] in ("top2sum_zero", "per_group") and Eh != Et:
        # round 3, batch 3a: a 0-filled group limit (group_mask_zero_fill) and Intern-S1-Pro's per-group top-(k / G)
        # (router_group_balanced) are played by their own code with the target's experts only: extras / copies need
        # theorems moeR_group_copies at fill 0 and moeR_per_group_copies (a -inf target on a 0-fill host by a uniform
        # selection-bias shift: group_zero_fill_shift -- walled above by the group-code compare)
        return f"{T['group_cap']} groups: host experts {Eh} != {Et}"
    if T["n_group"] and Eh % (m * T["n_group"]):
        return f"experts {Eh} not in {m} x {T['n_group']} equal groups"
    return None

def _rec_layer(x):
    """A per-layer type name that is a recurrent mixer (mamba / linear / delta / conv / ...)
    rather than a softmax-attention layer ('sliding_attention', 'deepseek_sparse_attention')."""
    s = str(x).lower()
    if not any(r in s for r in _REC_LAYER_NAMES):
        return False
    softmax_name = ("attention" in s or "attn" in s) and not any(
        r in s for r in ("linear", "delta", "mamba", "ssm", "gdn", "kda"))
    return not softmax_name


def _all_attention_list(d):
    """An explicit per-layer type list (layers_block_type / layer_types / hybrid_override_pattern) naming
    only softmax-attention (and FFN) layers, in a model with no parallel recurrent branch: the hybrid
    model class run as a plain softmax stack."""
    if str(d.get("model_type") or "").lower() in ("falcon_h1", "hymba") or d.get("pb_hybrid_layers"):
        return False
    for key in ("layers_block_type", "layer_types", "hybrid_override_pattern"):
        v = d.get(key)
        seq = list(v) if isinstance(v, str) else v
        if isinstance(seq, list) and seq and all(isinstance(x, str) for x in seq):
            kinds = [_kind(x) for x in seq]
            return all(k in ("full", "local", "ffn") for k in kinds) and any(k in ("full", "local") for k in kinds)
    return False


def _kv_source_shared(d):
    """Cross-layer KV sharing: a per-layer source map where some layer reads another layer's K,V
    (Mistral-Large-4 kv_source_layers), or the list of KV-producing layers that the
    'shared_compressed_attention' layers read from (DeepSeek-V4.1 kv_source_layer_ids)."""
    v = d.get("kv_source_layers")
    if isinstance(v, list) and any(isinstance(x, int) and x != i for i, x in enumerate(v)):
        return True
    return bool(isinstance(d.get("kv_source_layer_ids"), list) and d["kv_source_layer_ids"])


def _block_experts(d):
    """gpt-oss-Puzzle: expert counts set PER BLOCK in block_configs (no top-level count) ->
    [(n_experts, sliding_window)] per block, or []."""
    bc = d.get("block_configs")
    if not isinstance(bc, list):
        return []
    return [(_num(b.get("num_local_experts")), b.get("sliding_window")) for b in bc
            if isinstance(b, dict) and _num(b.get("num_local_experts"))]


def _k_eq_v(d):
    """Gemma-4 attention_k_eq_v: the FULL (non-sliding) attention layers have no v_proj, V is the raw
    k_proj output (v = v_norm(k_proj x), modeling_gemma4.py Gemma4TextAttention.use_alternative_attention).
    Round 3 (k_eq_v_by_code): DiffusionGemma's config class deletes the key (modular_diffusion_gemma.py:101) and its
    code builds v_proj only on the sliding layers (:258-262 / :360-364, value_states = key_states otherwise, :293 /
    :395): Gemma-4's K = V path fixed in code (K_EQ_V_BY_CODE)."""
    return bool(d.get("attention_k_eq_v")) or str(d.get("model_type") or "").lower() in K_EQ_V_BY_CODE


K_EQ_V_BY_CODE = _R3Set((), add={"k_eq_v_by_code": {"diffusion_gemma", "diffusion_gemma_text"}})


def _k_eq_v_all(d):
    """DeepSeek-V4: ONE shared head per layer, kv_proj -> kv_norm, the same tensor is K and V on EVERY layer
    (modeling_deepseek_v4.py; V4.1 assumed the same)."""
    return str(d.get("model_type") or "").lower() in VNORM_TYPES


_KV_STD_KEYS = ("num_key_value_heads", "num_kv_heads", "n_head_kv", "num_attention_groups",
                "num_key_value_heads_per_layer", "multi_query")


def _nkv_alias(d):
    """Round 3 (n_kv_heads_key_unread): the KV-head count under n_kv_heads / n_kv_head when no standard key is set (Nae1:
    Nae1GQA k / v projections n_embd -> n_kv_heads * head_dim, repeat_interleave(n_query_groups), nae1_model.py:112-146;
    nanochat / gaggle's n_kv_head = n_head), else 0."""
    if not _r3("n_kv_heads_key_unread") or any(d.get(k) not in (None, False) for k in _KV_STD_KEYS):
        return 0
    return _num(d.get("n_kv_heads") or d.get("n_kv_head"))


def _attn_variants(d, tagged=False):
    """Per-layer attention shapes beyond the top-level one, as (heads, qk_head_dim, kv_cap)
    (tagged=True: (heads, qk_head_dim, kv_cap, tied) with tied = the shape belongs to K = V layers).

    Heterogeneous stacks keep a second attention shape under its own keys: Gemma-4's global
    layers (global_head_dim / num_global_key_value_heads), a per_layer_config override map,
    sliding-window layers with their own MLA or GQA shape (swa_kv_lora_rank / swa_qk_*_head_dim,
    swa_head_dim / swa_num_*, Step-3.5 attention_other_setting). The host must hold the WIDEST.
    Gemma-4's global shape is the K = V one when attention_k_eq_v; num_global_key_value_heads is read
    only then (Gemma4TextConfig: 'attention_k_eq_v gates the kv-head override')."""
    out = []
    nh = _num(d.get("num_attention_heads") or d.get("n_head"))
    nkv = _num(d.get("num_key_value_heads")) or _num(d.get("num_attention_groups")) or _nkv_alias(d) or nh
    keqv = _k_eq_v(d)
    aos = d.get("attention_other_setting")                 # Step-3.5: the sliding layers' shape
    if isinstance(aos, dict) and aos.get("head_dim"):
        hd = _num(aos["head_dim"])
        out.append((_num(aos.get("num_attention_heads")) or nh, hd,
                    (_num(aos.get("num_attention_groups")) or nkv) * hd, False))
    ghd = _num(d.get("global_head_dim"))
    if ghd:
        gkv = _num(d.get("num_global_key_value_heads")) if (keqv or "attention_k_eq_v" not in d) else 0
        out.append((nh, ghd, (gkv or nkv) * ghd, keqv))
    plc = d.get("per_layer_config")
    if isinstance(plc, dict):
        for v in plc.values():
            if isinstance(v, dict) and v.get("head_dim"):
                hd = _num(v["head_dim"])
                out.append((_num(v.get("num_attention_heads")) or nh, hd,
                            (_num(v.get("num_key_value_heads")) or nkv) * hd, keqv))
    sh = _num(d.get("swa_num_attention_heads")) or nh
    if d.get("swa_kv_lora_rank"):
        out.append((sh, _num(d.get("swa_qk_nope_head_dim")) + _num(d.get("swa_qk_rope_head_dim")),
                    _num(d["swa_kv_lora_rank"]) + _num(d.get("swa_qk_rope_head_dim")), False))
    elif d.get("swa_head_dim"):
        hd = _num(d["swa_head_dim"])
        out.append((sh, hd, (_num(d.get("swa_num_key_value_heads")) or nkv) * hd, False))
    em = _ernie_mla(d)
    if em:                    # round 3 (ernie_mla_layer_dims_unread): ERNIE-5.5's DSA / MLA layers, beside the HCA ones
        out.append((em["heads"], em["mla_nope"] + em["mla_rope"], em["kv_latent"] + em["mla_rope"], False))
    return out if tagged else [v[:3] for v in out]


def _ernie_mla(d):
    """Round 3 (ernie_mla_layer_dims_unread): ERNIE-5.5's MLA (DSA) layers -- compress_ratios[i] <= 0, is_dsa_layer
    (vLLM-ERNIE config.py:155-158) -- run Ernie5_5DSAAttention on the hybrid_mla_* dims (config.py:29-34, :82-87,
    :162-183; attention.py:780-829), not the top-level keys (the HCA layers' K = V head): their dims, Ernie5_5Config
    defaults for missing keys, else None."""
    if not _r3("ernie_mla_layer_dims_unread") or str(d.get("model_type") or "").lower() != "ernie5_5_moe":
        return None
    cr = d.get("compress_ratios")
    if not (isinstance(cr, list) and any(isinstance(x, (int, float)) and not isinstance(x, bool) and x <= 0
                                         for x in cr)):
        return None
    g = lambda k, dflt: _num(d.get(k)) if d.get(k) is not None else dflt
    return {"heads": g("hybrid_mla_num_attention_heads", 64), "q_latent": g("hybrid_mla_q_lora_rank", 1024),
            "kv_latent": g("hybrid_mla_kv_lora_rank", 512), "mla_nope": g("hybrid_mla_qk_nope_head_dim", 192),
            "mla_rope": g("hybrid_mla_qk_rope_head_dim", 64), "mla_v": g("hybrid_mla_v_head_dim", 256)}


def rope_fits(target, host):
    """The rot gate: can a host whose inv_freq tables rotate `host` dims per head (one entry per table)
    hold a target whose frequency schedules rotate `target` dims per head? Each target schedule's
    rotating pairs go into ONE host table (its frequencies set in the buffer), and a table shared by
    several target schedules holds all their (distinct) frequencies: an assignment of target schedules
    to host tables with sum(target dims on a table) <= that table's dims. Layer alignment (which host
    layers read which table) is not modelled -- the depth axes charge layers separately.

    NoPE layers in a RoPE mix (no_rope_layers, a per-layer-type rope_parameters entry of None:
    _rope_schedules) are a NEGATIVE entry -r: those layers need r dims per head that do NOT rotate,
    i.e. r zero-frequency slots in whichever table they read (a host table's frequencies can be set to
    0, as NoPE <= RoPE does) -- so a one-table full-RoPE host cannot rotate one layer and not the next
    unless its table has room for both. A negative HOST entry is a NoPE table (the host's own NoPE
    layers): it takes only zero-frequency (negative) target entries. 0 = no rotary dims at all (a
    NoPE target): needs nothing. With cap = need this is transitive: a packing of packings is one."""
    from itertools import product
    t = [r for r in target if r != 0]
    if not any(r > 0 for r in t):
        return True                                        # nothing rotates: a host sets inv_freq = 0
    hs = [r for r in host if r != 0]
    if not any(c > 0 for c in hs):
        return False
    for assign in product(range(len(hs)), repeat=len(t)):
        load = [0] * len(hs)
        okk = True
        for r, b in zip(t, assign):
            if r > 0 and hs[b] < 0:                        # a rotating schedule on a NoPE table
                okk = False
                break
            load[b] += abs(r)
        if okk and all(l <= abs(c) for l, c in zip(load, hs)):
            return True
    return False


def _is_alibi(d):
    """The attention runs ALiBi: an alibi key (Falcon, MPT attn_config, OLMo-1), BLOOM, or Baichuan-13B /
    Baichuan2-13B (the 7B models RoPE; the 13B configs carry model_max_length and no max_position_embeddings:
    modeling_baichuan.py of the 13B repos, _get_interleave ALiBi slopes)."""
    mt = str(d.get("model_type") or "").lower()
    return bool(d.get("alibi") or (d.get("attn_config") or {}).get("alibi") or mt == "bloom"
                or (mt == "baichuan" and "max_position_embeddings" not in d))


# ALiBi SLOPE SCHEMES (round 2, pass 2; literature review, legacy cross-cutting A6). The slopes are
# computed in code from the head count, so an ALiBi host plays an ALiBi target only where the target's slopes are
# among the host's (its extra heads zeroed; heads are permutable, so the ORDER does not matter):
#   BLOOM (build_alibi_tensor), MPT (gen_slopes: [1::2] + [::2] of the next power of two, a reorder), Baichuan-13B
#   (_get_interleave) compute the SAME real slope set S(n) (the ALiBi paper's schedule) -- checked numerically for
#   n = 8 ... 128 with the transformers / remote functions: they agree to <= 7e-7 relative, the fp32 rounding of
#   the different formulas. S(n) grows with n (S(n) is S(2^k) plus odd powers of S(2^(k+1)), monotone), so
#   S(n_t) <= S(n_h) whenever n_h >= n_t.
#   Falcon rounds its slopes to bf16 (modeling_falcon.py: slopes.bfloat16()): 1e-4 - 3e-3 relative off S(n) for
#   n >= 12, a different value set (equal only while every slope is a power of two, n <= 8).
#   MPT's remote llm-foundry code reads attn_config.alibi_bias_max B (slopes 2^(-k B / 2^ceil(log2 n))); the
#   native MptForCausalLM ignores it (B = 8).
# ALIBI_REL_TOL: two slopes are the same value within the fp32 rounding of the formulas, not within bf16's.
ALIBI_REL_TOL = 1e-6


def alibi_slopes(n, bias_max=8, bf16=False, scale=1.0):
    """The ALiBi slopes of n heads (MPT / llm-foundry form, which is BLOOM's / Baichuan's set at bias_max 8),
    sorted; bf16 = Falcon's bf16-rounded slopes; scale = a factor on every slope after the rounding (round 3:
    Falcon's 1 / sqrt(head_dim), falcon_alibi_slope_scale)."""
    import math
    import struct
    if n <= 0:
        return ()
    N = 1 << math.ceil(math.log2(n))
    s = [2.0 ** (-(k * bias_max / N)) for k in range(1, N + 1)]
    if N != n:
        s = (s[1::2] + s[::2])[:n]
    if bf16:
        def rb(x):                                         # fp32 -> bf16, round to nearest even
            b = struct.unpack("<I", struct.pack("<f", x))[0]
            b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
            return struct.unpack("<f", struct.pack("<I", b))[0]
        s = [rb(x) for x in s]
    if scale != 1.0:
        s = [x * scale for x in s]
    return tuple(sorted(s))


def alibi_nests(t, h):
    """Target ALiBi (n_heads, bias_max, bf16[, scale]) hosted by host ALiBi (n_heads, bias_max, bf16[, scale]): every
    target slope is a distinct host slope (within ALIBI_REL_TOL). Reflexive and transitive (a matching of matchings)."""
    T, H = alibi_slopes(*t), list(alibi_slopes(*h))
    j = 0
    for x in T:                                            # both sorted: a greedy merge finds a matching if any
        while j < len(H) and H[j] < x * (1 - ALIBI_REL_TOL):
            j += 1
        if j == len(H) or abs(H[j] - x) > ALIBI_REL_TOL * x:
            return False
        j += 1
    return True


def _alibi_dims(d):
    """dims of an ALiBi config: alibi_heads (0 = no ALiBi), alibi_max (the slope schedule's B), alibi_bf16
    (Falcon), alibi_off (1 = a shape-preserving config value removes the ALiBi bias: Falcon's alibi flag -- the
    model then runs RoPE, whose inv_freq a host sets to 0 for a NoPE target as any RoPE host does -- and the remote
    MPT's alibi_bias_max, which at a large value underflows every slope 2^(-k B / N) to exactly 0; BLOOM,
    Baichuan-13B and the native MptForCausalLM compute their slopes from the head count with no switch)."""
    if not _is_alibi(d):
        return dict(alibi_heads=0, alibi_max=0, alibi_bf16=0, alibi_off=0, alibi_scale=0.0)
    mt = str(d.get("model_type") or "").lower()
    n = _num(d.get("num_attention_heads") or d.get("n_head") or d.get("n_heads"))
    remote_mpt = mt == "mpt" and bool(d.get("auto_map"))
    bmax = (_num((d.get("attn_config") or {}).get("alibi_bias_max")) or 8) if remote_mpt else 8
    # round 3 (falcon_alibi_slope_scale): FalconModel.forward divides the ALiBi mask by sqrt(hidden_size // heads)
    # (tf5 modeling_falcon.py:780-785; the SDPA path, the default, adds it once: :380-387), so Falcon's slopes are
    # bf16(S(n)) / sqrt(head_dim); BLOOM / MPT / Baichuan-13B add S(n) unscaled
    hd = _num(d.get("hidden_size")) // n if n else 0
    sc = 1.0 / hd ** 0.5 if (_r3("falcon_alibi_slope_scale") and mt == "falcon" and hd) else 1.0
    return dict(alibi_heads=n, alibi_max=bmax, alibi_bf16=int(mt == "falcon"),
                alibi_off=int(mt == "falcon" or remote_mpt), alibi_scale=sc)


def _rope_layer_types(rp):
    """rope_parameters keyed by layer type ({'full_attention': {...}, 'sliding_attention': None}, DeepSeek-V4's
    {'main': {...}, 'compress': {...}}): {layer type: its parameters, None = no RoPE on those layers}."""
    if not isinstance(rp, dict):
        return {}
    return {k: v for k, v in rp.items()
            if isinstance(v, dict) or (v is None and ("attention" in k or k in ("main", "compress")))}


def _rope_schedules(d, outer=None):
    """The distinct RoPE frequency schedules of a config's attention layers: [(need, cap)] per schedule
    ([] = no rotary at all). A schedule is one inv_freq table: the main one, a per-layer-type
    rope_parameters entry, a local / sliding-window theta (Gemma-3 rope_local_base_freq, MiMo / dots3
    swa_rope_theta). need = the rotated dims per head it uses: qk_rope_head_dim (MLA's decoupled RoPE
    key, DeepSeek-V4), else rotary_dim, else partial_rotary_factor / rotary_pct (top level, or nested in
    rope_parameters) x the layer's head_dim (Gemma-4's full layers: global_head_dim). cap = the rotated
    dims per head its table can drive once the buffer is set (hf/partial_rope_host.py): = need for a
    sliced rotary (the head's first rot dims rotate, the rest pass), but the WHOLE head for Gemma-4's
    "proportional" RoPE, whose table spans the head with a zero (NoPE) tail.
    NoPE layers in a RoPE mix are a schedule of their own, (-need, -need) (rope_fits: zero-frequency
    slots): a per-layer-type entry of None (Beam), or no_rope_layers / no_rope_layer_interval /
    no_rope_layer_indices (SmolLM3, Flint, Mistral-Large-4: the FEATURES nope_layers)."""
    nh = _num(d.get("num_attention_heads") or d.get("n_head"))
    hidden = _num(next((d[k] for k in _LM_DIM_KEYS if d.get(k)), 0))
    hd = ((_num(d.get("head_dim")) if _stock_arch(d) not in _DERIVED_HEAD_DIM_ARCHS else 0)
          or (hidden // nh if nh else 0))
    rp = d.get("rope_parameters") if isinstance(d.get("rope_parameters"), dict) else {}
    f0 = _rotary_factor(d, outer)                          # (top level, nested, else the config class default)
    f0 = 1.0 if f0 is None else float(f0)
    qr = d.get("qk_rope_head_dim")
    mt = str(d.get("model_type") or "").lower()
    if _mla_off_rope(d):
        qr = None                                          # (round 3, use_mla_false_rope_dims: _mla_off_rope)
    rd = d.get("rotary_dim") if not (_r3("rotary_dim_ignored") and _rotary_dim_unread(d, outer)) else None
    zero_tail = _r3("rope_cap_zero_tail_table") and _zero_tail_rope(d, outer)

    def sched(f, head, kind):
        if qr is not None:
            need = _num(qr)
        elif rd:
            need = _num(rd)
        else:
            need = int(round(f * head))
        # (round 3, rope_cap_zero_tail_table: the remote LFM3 table spans the head with a zero tail, as "proportional")
        return need, (max(head, need) if kind == "proportional" or zero_tail else need)
    hs_l, pos_l = d.get("head_sizes"), d.get("position_layers")
    if (_r3("per_layer_head_shapes") and mt == "domynedge" and isinstance(hs_l, list) and isinstance(pos_l, list)
            and len(hs_l) == len(pos_l)):
        # round 3: Domyn-Edge's inv_freq spans config.rope_dim = the RoPE layers' head size (configuration_domynedge.py
        # :122-124, modeling_domynedge.py:87: 64), its NoPE full layers are 256 wide (head_sizes): (64, -256)
        rope_h = [_num(h) for h, p in zip(hs_l, pos_l) if str(p) == "rope"]
        nope_h = [_num(h) for h, p in zip(hs_l, pos_l) if str(p) != "rope"]
        if rope_h:
            return [sched(f0, max(rope_h), rp.get("rope_type") or rp.get("type"))] + (
                [(-max(nope_h), -max(nope_h))] if nope_h else [])
    per = _rope_layer_types(rp)
    rs = d.get("rope_scaling")
    lt = d.get("layer_types") if isinstance(d.get("layer_types"), list) else []
    if (_r3("local_rope_theta_unread") and mt == "olmo3" and not per and isinstance(rs, dict)
            and str(rs.get("rope_type") or rs.get("type") or "default") != "default"
            and any("sliding" in str(x) for x in lt) and any("full" in str(x) for x in lt)):
        # round 3: OLMo-3's config applies a legacy rope_scaling to rope_parameters["full_attention"] ONLY, the sliding
        # layers keep {"rope_type": "default"} (tf5 modular_olmo3.py:97-116), and the rotary is per layer type
        # (:222-223): the YaRN table on the 1-in-4 full layers, theta^(-2i/d) on the sliding ones -- two tables
        per = {"full_attention": dict(rs), "sliding_attention": {"rope_type": "default"}}
    out = {}
    pf_list = d.get("partial_rotary_factors")
    if _r3("partial_rope_keys_unread") and mt == "step3p5" and isinstance(pf_list, list) and pf_list and not per:
        # round 3: Step-3.5 sets config.partial_rotary_factor / rope_theta per layer (modeling_step3p5.py:56-59, :67)
        # and rotates the first cos.shape[-1] dims (:125): one table per distinct (factor, theta) -- 64 rotated dims
        # on its full layers (0.5, 5e6), 128 on its sliding ones (1.0, 1e4)
        th = d.get("rope_theta")
        th = th if isinstance(th, list) and len(th) == len(pf_list) else [th] * len(pf_list)
        for f, t in zip(pf_list, th):
            out[repr((float(f), str(t)))] = sched(float(f), hd, None)
        if _nope_layer_mix(d) and any(n > 0 for n, _ in out.values()):
            n0 = max(n for n, _ in out.values())
            out["nope"] = (-n0, -n0)
        return list(out.values())
    if per:
        ghd = _num(d.get("global_head_dim"))
        plc = d.get("per_layer_config")
        if not ghd and _r3("fp_gemma4_plc_rope") and isinstance(plc, dict) and isinstance(lt, list):
            # fr-positions: transformers-5 Gemma-4 configs carry the full layers' width as per_layer_config
            # {layer index: {"head_dim": 512}} instead of global_head_dim (Gemma4TextConfig.__post_init__ writes one
            # from the other, configuration_gemma4.py:210-222), and the rotary builds each layer type's inv_freq from
            # config.per_layer_config[layer_type] (modeling_gemma4.py:1103-1106): the full-attention table spans that
            # width (the per-layer head shapes already read it: _attn_variants)
            fw = [_num(v.get("head_dim")) for k, v in plc.items() if isinstance(v, dict) and v.get("head_dim")
                  and str(k).isdigit() and int(k) < len(lt) and "full" in str(lt[int(k)])]
            ghd = max(fw, default=0)
        for k, v in per.items():
            if v is None:                                  # NoPE layers of that type: zero-frequency slots
                n0 = sched(f0, ghd if (k == "full_attention" and ghd) else hd, None)[0]
                out["nope"] = (min(out.get("nope", (0, 0))[0], -n0),) * 2
                continue
            f = v.get("partial_rotary_factor")
            nc = sched(f0 if f is None else float(f), ghd if (k == "full_attention" and ghd) else hd,
                       v.get("rope_type") or v.get("type"))
            key = repr(sorted((kk, str(vv)) for kk, vv in v.items() if kk != "partial_rotary_factor"))
            out[key] = max(out.get(key, (0, 0)), nc)
    else:
        out["main"] = sched(f0, hd, rp.get("rope_type") or rp.get("type"))
    theta = d.get("rope_theta") or rp.get("rope_theta") or d.get("rotary_emb_base")
    for k in ("rope_local_base_freq", "swa_rope_theta", "compress_rope_theta"):   # (DeepSeek-V4: compress layers)
        if d.get(k) and d[k] != theta and not per:
            out[k] = out["main"]
    if _r3("local_rope_theta_unread") and mt == "plamo3" and not per and (
            d.get("window_size") or any("sliding" in str(x) for x in lt)):
        # round 3: PLaMo-3 builds its rope_parameters in code (modeling_plamo.py:212-240): the sliding layers
        # rope_local_theta (default 1e4, default type), the full layers rope_theta (YaRN when rope_scaling_factor != 1)
        loc, glob = d.get("rope_local_theta", 10000), d.get("rope_theta", 10000)
        try:
            yarn = float(d.get("rope_scaling_factor") or 1.0) != 1.0
        except (TypeError, ValueError):
            yarn = False
        if loc != glob or yarn:
            out["rope_local_theta"] = out["main"]
    if _nope_layer_mix(d) and "nope" not in out and any(n > 0 for n, _ in out.values()):
        n0 = max(n for n, _ in out.values())
        out["nope"] = (-n0, -n0)
    return list(out.values())


def _nope_layer_mix(d):
    """RoPE on some layers and NoPE on others, by a per-layer switch (no_rope_layers: 1 = RoPE, 0 = NoPE;
    no_rope_layer_interval; no_rope_layer_indices) or a per-layer-type rope_parameters entry of None."""
    nrl = d.get("no_rope_layers")
    if ((isinstance(nrl, list) and len(set(nrl)) > 1) or _num(d.get("no_rope_layer_interval")) > 1
            or (isinstance(d.get("no_rope_layer_indices"), list) and d["no_rope_layer_indices"])):
        return True
    per = _rope_layer_types(d.get("rope_parameters"))
    return any(v is None for v in per.values()) and any(isinstance(v, dict) for v in per.values())


def _nope_flags(d, n):
    """fr-positions (fp_nope_layer_host): which of a RoPE + NoPE mix's n layers run NoPE (True), from the switch the
    code reads -- no_rope_layers (1 = RoPE, 0 = NoPE: SmolLM3 / Llama-4 use_rope = no_rope_layers[i]; _normalize writes
    it for afmoe / Cohere-2 / EXAONE-4 / EXAONE-MoE / Kolibri / Muse / Domyn-Edge / Grug), no_rope_layer_indices
    (Flint), no_rope_layer_interval (SmolLM3's default list: NoPE where (i + 1) % interval == 0), or a per-layer-type
    rope_parameters entry of None (Beam) over layer_types -- else None (not a mix, or not locatable)."""
    if not _nope_layer_mix(d):
        return None
    nrl = d.get("no_rope_layers")
    if isinstance(nrl, list) and nrl:
        return [not bool(_num(x)) for x in nrl]
    idx = d.get("no_rope_layer_indices")
    if isinstance(idx, list) and idx and n:
        s = {int(_num(x)) for x in idx}
        return [i in s for i in range(n)]
    iv = _num(d.get("no_rope_layer_interval"))
    if iv > 1 and n:
        return [(i + 1) % iv == 0 for i in range(n)]
    per = _rope_layer_types(d.get("rope_parameters"))
    lt = d.get("layer_types")
    if per and isinstance(lt, list) and lt:
        return [str(x) in per and per[str(x)] is None for x in lt]
    return None


_LOCAL_MARKS = ("sliding", "swa", "local", "chunk", "compress", "window")
_FFN_ENTRIES = {"-", "e", "mlp", "moe", "ffn"}


def _kind(x):
    """Per-layer entry -> 'full' / 'local' (softmax attention), 'rec' (SSM / linear / conv), 'ffn'
    (a separate FFN sub-layer entry, Nemotron-H / Dory), or None (unknown)."""
    s = str(x).lower()
    if s in _FFN_ENTRIES:
        return "ffn"
    if s == "*":
        return "full"
    if s in ("m", "recurrent") or _rec_layer(s):
        return "rec"
    if any(m in s for m in _LOCAL_MARKS):
        return "local"
    if s in ("t", "a", "h", "hybrid") or any(m in s for m in ("attention", "attn", "full", "global")):
        return "full"                                   # incl. DSA 'indexed' / 'sparse' (classify: full)
    return None


def _attn_counts(d, n, only=None):
    """Layers of a model that a PLAIN softmax target can use (the ZERO_EDGES depth currency):
    (attn, full, base, shared_cap) -- `attn` own-KV softmax-attention layers (any window), `full` of
    them with a full (global) window, counted over `base` listed layer slots; shared_cap = the number
    of DISTINCT attention blocks when one block is shared across many positions (Zamba), else None.

    Read from the per-layer lists / indices each schema uses; a model whose layout cannot be read
    gets 0 for the kind it cannot prove (conservative: no zeroing edge).
    only (fr-positions, fp_nope_layer_host): a per-layer bool list; the kind counts kc then count only the layers it
    marks (None when its length is not the layer list's: the caller treats the split as unknown)."""
    mt = str(d.get("model_type") or "").lower()
    g = lambda k: _num(d.get(k))
    sw_on = bool(d.get("sliding_window") or (d.get("attn_config") or {}).get("sliding_window")) \
        and d.get("use_sliding_window", True)
    kinds, shared = None, None
    pat = d.get("hybrid_override_pattern")
    bc = d.get("block_configs")
    if mt == "nemotron-nas" and isinstance(bc, list) and bc:
        # DeciLM / Nemotron-NAS: per block, the attention may be a no-op or a token-wise linear map
        # (replace_with_linear): such a block has no attention to play a target's attention layer.
        # Blocks with BOTH halves no-op are not layers at all (dims_of counts the live blocks).
        kinds = []
        for b in bc:
            a, f = (b.get("attention") or {}), (b.get("ffn") or {})
            if a.get("no_op") and f.get("no_op"):
                continue
            kinds.append("noattn" if (a.get("no_op") or a.get("replace_with_linear"))
                         else ("local" if a.get("window_length") else "full"))
    if mt == "hymba" and n:                                  # attention + SSM heads in EVERY layer;
        gi = set(d.get("global_attn_idx") or [])             # full attention only on global_attn_idx
        kinds = [("full" if (not sw_on or i in gi) else "local") for i in range(n)]
    for key in (() if kinds else ("layers_block_type", "layer_types", "layer_type")):
        v = d.get(key)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            kinds = [_kind(x) for x in v]
            break
    if isinstance(pat, str) and pat and (kinds is None or mt in NEMOTRON_H_TYPES):
        kinds = [_kind(c) for c in pat]
    if kinds is None:
        for key, soft in (("attn_type_list", 1), ("linear_attention_pattern", 0)):   # MiniMax / Youtu
            v = d.get(key)
            if isinstance(v, list) and v:
                kinds = ["full" if x == soft else "rec" for x in v]
        hlp = d.get("hybrid_layer_pattern")                 # MiMo-V2: 0 = full, 1 = sliding window
        if kinds is None and isinstance(hlp, list) and hlp and all(x in (0, 1) for x in hlp):
            kinds = ["full" if x == 0 else "local" for x in hlp]
    if kinds is None:
        bt = d.get("_block_types") or d.get("block_types")  # RecurrentGemma: a repeating pattern
        if isinstance(bt, list) and bt and n:
            kinds = [_kind(bt[i % len(bt)]) for i in range(n)]
            kinds = ["local" if k == "full" and d.get("attention_window_size") else k for k in kinds]
    if kinds is None and n:
        idx = None
        lac = d.get("linear_attn_config")
        if isinstance(lac, dict) and isinstance(lac.get("full_attn_layers"), list):
            idx = len(lac["full_attn_layers"])               # Kimi-Linear (KDA + periodic full MLA)
        for key in ("full_attn_idxs", "attn_layer_indices", "full_attention_layers"):
            if idx is None and isinstance(d.get(key), list):
                idx = len(d[key])
        if idx is None and g("attn_layer_period"):           # Jamba / Zamba-v1
            per, off = g("attn_layer_period"), g("attn_layer_offset")
            idx = sum(1 for i in range(n) if i % per == off)
        if idx is None and g("full_attention_interval") and mt not in ("plamo2", "plamo2vl"):
            idx = n // g("full_attention_interval")
        if idx is None and g("gdn_layers_per_attention"):    # GPT-S3: k GDN layers per softmax layer
            idx = n // (g("gdn_layers_per_attention") + 1)
        if idx is not None:
            kinds = ["full"] * idx + ["rec"] * (n - idx)
            if _r3("kind_order"):
                # round 3 (kind_order): the layers IN ORDER, as each family's code lists them -- Kimi-Linear
                # is_kda_layer reads (layer_idx + 1) in kda_layers (1-based; configuration_kimi.py:136), LFM2
                # layer_types = full_attention iff i in full_attn_idxs (configuration_lfm2.py), Bamba
                # layers_block_type iff i in attn_layer_indices, Jamba i % attn_layer_period == attn_layer_offset
                # (configuration_jamba.py), Qwen3-Next full iff (i + 1) % full_attention_interval == 0
                # (configuration_qwen3_next.py), GPT-S3 is_attention_layer (i + 1) % (gdn_layers_per_attention + 1)
                # == 0 (configuration_gpts3.py:48, read only); the counts are unchanged
                at = None
                if isinstance(lac, dict) and isinstance(lac.get("full_attn_layers"), list):
                    at = {_num(x) - 1 for x in lac["full_attn_layers"]}
                for key in ("full_attn_idxs", "attn_layer_indices", "full_attention_layers"):
                    if at is None and isinstance(d.get(key), list):
                        at = {_num(x) for x in d[key]}
                if at is None and g("attn_layer_period"):
                    at = {i for i in range(n) if i % per == off}
                if at is None and g("full_attention_interval") and mt not in ("plamo2", "plamo2vl"):
                    at = {i for i in range(n) if (i + 1) % g("full_attention_interval") == 0}
                if at is None and g("gdn_layers_per_attention"):
                    at = {i for i in range(n) if (i + 1) % (g("gdn_layers_per_attention") + 1) == 0}
                if at is not None and len(at & set(range(n))) == idx:
                    kinds = ["full" if i in at else "rec" for i in range(n)]
        elif mt in ("plamo2", "plamo2vl") and g("mamba_step"):   # attention where i % step == step // 2
            st = g("mamba_step")
            fi = set(d.get("full_attention_idx") or [])
            kinds = [("full" if i in fi else "local") if i % st == st // 2 else "rec" for i in range(n)]
        elif mt == "falcon_h1" or d.get("pb_hybrid_layers"):
            # parallel attention + SSM in EVERY layer (Falcon-H1, PseudoBrain's parallel GDN)
            kinds = [("local" if sw_on else "full")] * n
        elif mt in HYBRID_TYPES or mt == "phi4flash":
            kinds = ["rec"] * n                              # layout not readable: no zeroing edge
        elif mt in SSM_TYPES | LINEAR_TYPES:
            # a pure SSM / linear stack (Mamba-1, Falcon-Mamba, RWKV, GLA): every layer recurrent, no attention
            # (round 2, pass 5, verify/REVIEW.md finding 2: it fell through to the softmax default below)
            kinds = ["rec"] * n
    if kinds is None:                                        # a softmax stack without a per-layer list
        if not sw_on:
            kinds = ["full"] * n
        elif mt == "gemma2":                                  # Gemma-2: odd layers global (code-checked)
            kinds = ["local" if i % 2 == 0 else "full" for i in range(n)]
        elif g("sliding_window_pattern") or mt in ("gemma3", "gemma3_text"):
            p = g("sliding_window_pattern") or 6             # Gemma-3 / Cohere2 / EXAONE-4: every p-th global
            kinds = ["full" if (i + 1) % p == 0 else "local" for i in range(n)]
        elif mt in ("qwen2", "qwen3", "qwen2_moe", "qwen3_moe") and d.get("max_window_layers") is not None:
            kinds = ["local" if i >= g("max_window_layers") else "full" for i in range(n)]
        else:
            kinds = ["local"] * n                            # sliding window on every layer
    # own-KV filter (cross-layer KV sharing): a layer that reads another layer's K,V can't host one
    own = [True] * len(kinds)
    nsh = g("num_kv_shared_layers")
    if nsh:
        own = [i < len(kinds) - nsh for i in range(len(kinds))]
    ksl = d.get("kv_source_layers")
    if isinstance(ksl, list) and len(ksl) == len(kinds):
        own = [x is None or x == i for i, x in enumerate(ksl)]
    if d.get("use_cla") and _num(d.get("cla_share_factor")) > 1:   # Hunyuan CLA: every f-th layer computes K,V
        own = [o and i % _num(d["cla_share_factor"]) == 0 for i, o in enumerate(own)]
    if isinstance(d.get("kv_reuse_group"), list):            # Hymba: consecutive layers share K,V
        for grp in d["kv_reuse_group"]:
            for j in (grp[1:] if isinstance(grp, list) else []):
                if isinstance(j, int) and j < len(own):
                    own[j] = False
    if isinstance(d.get("kv_source_layer_ids"), list) and d["kv_source_layer_ids"]:
        lt = d.get("layer_types")
        if isinstance(lt, list) and len(lt) == len(kinds):
            own = [o and "shared" not in str(x) for o, x in zip(own, lt)]
        else:                                                # which layers share is not listed
            src = set(d["kv_source_layer_ids"])
            own = [o and i in src for i, o in enumerate(own)]
    if mt in ("zamba", "zamba2"):                            # ONE (Zamba) / num_mem_blocks shared block(s)
        shared = g("num_mem_blocks") or 1
    # Nemotron-H-style lists name the FFN separately: an attention entry hosts a target layer only
    # if an FFN entry follows it before the next attention entry
    sep_ffn = "ffn" in kinds
    keqv, keqv_all = _k_eq_v(d), _k_eq_v_all(d)
    kc = dict.fromkeys(LAYER_KINDS, 0)
    if only is not None and len(only) != len(kinds):
        return None
    toks = []                                                # (round 3, kind_order: (layer index, kind), in order)
    for i, k in enumerate(kinds):
        if only is not None and not only[i]:
            continue
        if k == "rec":
            kc["R"] += 1
            toks.append((i, "R"))
            continue
        if k not in ("full", "local"):
            continue
        if sep_ffn:
            nxt = next((kk for kk in kinds[i + 1:] if kk in ("full", "local", "ffn")), None)
            if nxt != "ffn":
                continue
        if not own[i]:
            kc["S"] += 1                                     # reads another layer's K,V (CLA)
            toks.append((i, "S"))
            continue
        tied = (keqv and k == "full") or keqv_all            # Gemma-4: K = V on the non-sliding layers; V4: all
        kk = ("F" if k == "full" else "L") + ("T" if tied else "U")
        kc[kk] += 1
        toks.append((i, kk))
    # parallel hybrids: a recurrent mixer beside the attention in the layer (Falcon-H1, Hymba: every
    # layer; PseudoBrain pb_hybrid_layers: the listed ones). Round 3 (kind_order): such a layer is ONE position of
    # the layer sequence, kind "X|R" (attention kind X and a recurrent mixer side by side on the same normed input):
    # it plays a target layer of kind X (zero the recurrent mixer's out_proj), R (zero o_proj) or Y|R, never two
    # sequential target layers (the second reads the stream after the first's update)
    par = set()
    if mt in ("falcon_h1", "hymba"):
        kc["R"] = max(kc["R"], len(kinds))
        par = set(range(len(kinds)))
    elif isinstance(d.get("pb_hybrid_layers"), list):
        kc["R"] = max(kc["R"], len(d["pb_hybrid_layers"]))
        par = {_num(x) for x in d["pb_hybrid_layers"]}
    toks = [t + "|R" if (i in par and t != "R") else t for i, t in toks]
    attn = kc["FU"] + kc["FT"] + kc["LU"] + kc["LT"]
    full = kc["FU"] + kc["FT"]
    base = sum(1 for k in kinds if k != "ffn") if (sep_ffn and mt not in NEMOTRON_H_TYPES) else len(kinds)
    return attn, full, max(base, 1), shared, kc, toks


# Per-layer KINDS (the ZERO_EDGES generalised to every target): a target layer of kind t can only be
# played by a host layer of a kind that reaches it --
#   FU  full-window attention, own K and V      FT  full-window, K = V (Gemma-4 attention_k_eq_v)
#   LU  sliding-window attention, own K and V   LT  sliding-window, K = V
#   S   attention reading another layer's K,V (cross-layer KV sharing; the CLA carry is not payable)
#   R   a recurrent (SSM / linear / conv) mixer sublayer (the near-gap is not payable)
# full reaches local (mask it down), own-V reaches K = V (copy k_proj into v_proj: a separate-V host
# plays a K = V layer exactly), nothing reaches across S / R. emul_matrix requires a perfect matching of
# the target's layers into the host's (Hall's condition over every set of target kinds), on the unrolled
# and on the weight-distinct counts.
LAYER_KINDS = ("FU", "FT", "LU", "LT", "S", "R")
KIND_REACH = {"FU": ("FU", "FT", "LU", "LT"), "FT": ("FT", "LT"), "LU": ("LU", "LT"), "LT": ("LT",),
              "S": ("S",), "R": ("R",)}                      # host kind -> target kinds it can play


def reach(order):
    """Transitive closure: value -> set of all values it is <= (incl. itself)."""
    nodes = set(order) | {p for v in order.values() for (p, _, _) in v}
    R = {n: {n} for n in nodes}
    changed = True
    while changed:
        changed = False
        for v, parents in order.items():
            for (p, _, _) in parents:
                if not R[p] <= R[v]:
                    R[v] |= R[p]
                    changed = True
    return R


def one_hop(order):
    """value -> {itself} + its DIRECT parents: the non-transitive reach used for the activation
    slot, whose floors compound under composition (a fit of a fit is not a fit at the same floor)."""
    nodes = set(order) | {p for v in order.values() for (p, _, _) in v}
    return {n: {n} | {p for (p, _, _) in order.get(n, [])} for n in nodes}


REACH = {name: (one_hop(order) if name == "act" else reach(order)) for name, order in SLOTS}


def branches(v):
    """A host slot value may be a BRANCH SET 'a|b|c' (the universal host carries several variants
    natively, e.g. act 'SiLU|situ|relu2'); a plain value is a one-element set."""
    return v.split("|") if isinstance(v, str) and "|" in v else [v]


def slot_reaches(slot, a, b, strict=False):
    """Target value a is hosted by host value b in `slot` (b may be a branch set). strict: only
    through EXACT reductions."""
    orders = dict(SLOTS)
    for bb in branches(b):
        if all(_reach1(slot, orders[slot], aa, bb, strict) for aa in branches(a)):
            return True
    return False


def _regime_strict(slot):
    """A regime slot whose approximate edges are measured NOT payable (crossed only exactly). (The qk-norm /
    softcap slots are priced edge by edge instead: REGIME_PAYABLE, regime_reaches.)"""
    return False                       # (the sink slot is a regime slot since round 2, pass 5: REGIME_PAYABLE)


# the regime slots' reach over their PAYABLE edges only: exact edges, and the approx ones REGIME_PAYABLE pays
# (a path through an unpayable regime edge does not count: head -> no -> proj is not a payable qk-norm crossing)
PAYABLE_REACH = {name: reach({v: [(p, c, ex) for (p, c, ex) in edges if ex or REGIME_PAYABLE.get((name, v, p), False)]
                              for v, edges in order.items()})
                 for name, order in SLOTS if name in REGIME_SLOTS}


def regime_reaches(slot, a, b):
    """slot_reaches for a qk-norm / softcap REGIME slot under the headline: a crossing counts only along
    edges that are exact or measured payable (REGIME_PAYABLE); b may be a branch set."""
    return any(all(aa == bb or bb in PAYABLE_REACH[slot].get(aa, {aa}) for aa in branches(a)) for bb in branches(b))


def payable_reaches(slot, a, b):
    """slot_reaches under the HEADLINE: as slot_reaches, but the mixer near-gap counts only if
    MIXER_GAP_PAYABLE (it is measured not payable), the CLA carry only if CLA_CARRY_PAYABLE, the
    qk-norm / softcap regimes only in the directions REGIME_PAYABLE records as payable, and the
    sink-on-a-plain-host edge only if measured payable."""
    if slot in REGIME_SLOTS:
        return regime_reaches(slot, a, b)
    return slot_reaches(slot, a, b, strict=(slot == "mixer" and not MIXER_GAP_PAYABLE)
                        or (slot == "crosslayer_kv" and not CLA_CARRY_PAYABLE) or _regime_strict(slot))


def _reach1(slot, order, a, b, strict):
    if a == b:
        return True
    if b not in REACH[slot].get(a, {a}):
        return False
    if not strict:
        return True
    if slot == "act":                                  # one-hop: the direct edge's exactness
        return any(p == b and ex for (p, _, ex) in order.get(a, []))
    return (edge_cost(order, a, b) or (None, False))[1]


def edge_cost(order, a, b):
    """currency + exactness to go a -> b (a<=b) in a slot order; None if a==b or unreachable.
    The shortest ALL-EXACT path if one exists, else the shortest path (the orders have 2-cycles:
    sink / mtp free both ways, LayerNorm <-> RMSNorm, the qk-norm / softcap regime edges)."""
    if a == b:
        return None
    from collections import deque
    for exact_only in (True, False):                  # BFS over the cover graph, visited-set guarded
        q, seen = deque([(a, [], True)]), {a}
        while q:
            cur, curr, exact = q.popleft()
            if cur == b:
                return (curr, exact)
            for (p, c, ex) in order.get(cur, []):
                if (ex or not exact_only) and p not in seen:
                    seen.add(p)
                    q.append((p, curr + [c], exact and ex))
    return None


# --- forward-pass FEATURES no slot carries (the edge-case audit: dist/edge-case-audit.md) ----------------
# Each is a branch of the universal host (elephantine/universal.py) that some family switches on and the
# slots above do not see. Per feature: (target side, host side, why) --
#   target side  can a host WITHOUT the feature play a target WITH it?
#   host side    can a host WITH the feature play a target WITHOUT it (switch it off)?
# "free" = an exact weight setting; "wall" = no currency (no exact construction, and either measured not
# payable or a per-token op of the same kind as one measured not payable: block/fold_agreement.py).
# The feature set of a model is classify()'s "feats" (a '+'-joined string); emul_matrix requires, in every
# mode but "capacity", each target feature to be carried by the host or "free" on the target side, and each
# host feature to be carried by the target or "free" on the host side.
FEATURES = {
    "tied_emb":   ("free", "wall",
                   "tied embeddings (lm_head = E^T): a tied target is an untied host with lm_head := E^T; a tied "
                   "host's output table IS its input table -- a target with its own lm_head would need a second "
                   "table in the residual, which every pre-norm sees (block/prenorm_carry.py)"),
    "out_gate":   ("wall", "free",
                   "attention output gate o * sigmoid(W_g x) (Qwen3-Next / 3.5 attn_output_gate, Laguna, Step-3.5, "
                   "MiniCPM-SALA, Kimi MLA, ...): a data-dependent product no ungated layer forms; a gated host sets "
                   "W_g = 0 (gate 1/2) and doubles o_proj"),
    "switch_lora": ("wall", "free",
                    "per-token switched LoRA adapters (Granite-Switch aLoRA): the weights change mid-sequence on a "
                    "control token; a plain host has one weight set (a switch host never selects an adapter)"),
    "hyper_conn": ("wall", "wall",
                   "manifold-constrained hyper-connections (hc_mult residual streams, Sinkhorn-mixed: DeepSeek-V4, "
                   "GLM-5-Next, Qwen4-exp; round 3: needle / SAN mhc_lanes, and a variant key hc_<code> each): one residual stream cannot carry N mixed ones under a pre-norm; the "
                   "Sinkhorn comb reaches the identity only as a limit and the pre / post maps are sigmoids"),
    "engram":     ("wall", "free",
                   "hashed n-gram memory added to the residual (Engram, DeepSeek-V4.1): a token embedding cannot "
                   "read the previous tokens; zero its output to switch it off"),
    "ple":        ("wall", "free",
                   "per-layer token embeddings fed into every layer (Gemma-3n / Gemma-4 E-models PLE; round 3: Qwen4-exp's is "
                   "an n-gram memory, engram_qwen4): "
                   "carrying them in the residual is seen by every pre-norm; zero them to switch off"),
    "value_emb":  ("wall", "free",
                   "value embeddings / value residuals added inside attention (nanochat ve, x0 / residual lambdas, "
                   "smear): extra per-token inputs no plain layer reads; zero them to switch off"),
    "attn_res":   ("wall", "wall",
                   "AttnRes depth-softmax residuals (Kimi): the residual is a softmax-weighted mix of earlier "
                   "block outputs, not their sum; a plain residual is not a softmax mix"),
    "vnorm":      ("wall", "wall",
                   "per-head RMS norm of V (Gemma-4 v_norm, weight-free): a per-token normalisation of the values "
                   "-- the same fold as a qk-norm, measured not payable (block/fold_agreement.py); weight-free, so "
                   "a v-norm host cannot switch it off"),
    "bias_qkv":   ("wall", "free",
                   "q / k / v projection biases (Qwen-1 / 2 / 2.5, GPT-2, OPT, Bloom, Pythia, Phi, StableLM, ...): a "
                   "constant needs a constant input channel, which a pre-norm residual does not offer (RMSNorm "
                   "rescales it per token); dropping them is measured not payable (Qwen2.5-0.5B: q 7%, k 30% -- "
                   "RoPE rotates the key bias, so it is not softmax-invariant -- v 0% top-1, block/fold_agreement.py)"),
    "bias_out":   ("wall", "free",
                   "output-side biases: o_proj / MLP biases and affine LayerNorm beta (folded into the consumers' "
                   "biases and an lm_head bias: the LayerNorm -> RMSNorm fold needs them, tests/"
                   "test_cookbook_fixes.py): a bias-free host has no constant to add"),
    "post_norm":  ("wall", "wall",
                   "SANDWICH norms: a pre-norm AND a norm on each sublayer's OUTPUT before the residual add (Gemma-2 "
                   "/ 3 / 4, GLM-4-0414, AFMoE, HyperCLOVA X, PLaMo-2, Muse, ...): the output norm folded to a constant "
                   "is measured not payable (OLMo-2-1B: top-1 14%, block/fold_agreement.py), and a sandwich host "
                   "cannot switch its norm off"),
    # round 2, pass 2: post-ONLY norms split from sandwich ones (literature review: OLMo-2 / 3, FlexOlmo, OLMo-hybrid
    # attention layers and EXAONE-4 / 4.5 have no pre-norm). The sublayers of a post-only layer read the UN-normalised
    # residual: a sandwich host cannot remove its pre-norm (it rescales the input per token) and a post-only host
    # cannot add one; neither plays a plain pre-norm target or the other. Same rule both ways as post_norm.
    "post_only":  ("wall", "wall",
                   "post-ONLY norms: no pre-norm, a norm on each sublayer's output (OLMo-2 / 3, FlexOlmo, OLMo-hybrid "
                   "attention layers, EXAONE-4 / 4.5): the sublayers read the un-normalised residual, which a pre-norm "
                   "host (plain or sandwich) cannot give them; a post-only host has no pre-norm to play one"),
    # round 2, pass 3: the router's scoring, renormalisation and group limit are compared by emul_matrix's MoE rules
    # (moe_why on the dims moe_sig: the score / renorm / group-score each code-fixed or a config value per family),
    # not as features; these three stay in the feature string for the record and are free here
    "router_sigmoid": ("free", "free",
                       "sigmoid expert scores (DeepSeek-V3, GLM-4.5, Kimi-K2, ...): priced by moe_why (router score): a "
                       "softmax router's weights cannot equal sigmoid ones for linear logits, and the sigmoid -> e^z "
                       "limit is a limit with no measured agreement, so a host needs the target's score in its code "
                       "or its config"),
    "router_bias": ("wall", "free",
                    "a selection-only expert bias (aux-loss-free balancing: topk_method noaux_tc, e_score_correction_"
                    "bias, moe_router_enable_expert_bias, use_routing_bias, ...): the experts are SELECTED on l + b but "
                    "weighted by l; a bias-free router selects on its weighting logits (Lean router_bias_not_bias_free); "
                    "a biased host sets b = 0"),
    "shared_gate": ("wall", "free",
                    "a sigmoid gate on the shared expert, sigmoid(w . x) * shared(x) (Qwen1.5 / 2-MoE, Qwen3-Next, "
                    "Qwen3.5-MoE, Qwen4-exp): a data-dependent product an ungated shared expert does not form; a gated "
                    "host sets w = 0 (gate 1/2) and doubles the shared down-projection"),
    "router_group": ("free", "free",
                     "group-limited top-k (n_group / topk_group): priced by moe_why (group routing: the same group-"
                     "score code, m = 1 under a top-2-sum score, the host's experts in m x n_group equal groups)"),
    "router_in_gate": ("wall", "wall",
                       "the router score scales the expert INPUT (Llama-4: routed_in = x * sigmoid(l)): f(s x) != s f(x) "
                       "for a SwiGLU expert, so an output-gated router cannot play it, and an input-gated host cannot "
                       "give a target its output gate"),
    "nope_layers": ("wall", "wall",
                    "RoPE on some layers and NoPE on others (no_rope_layers / no_rope_layer_interval / "
                    "no_rope_layer_indices: SmolLM3, Flint, Mistral-Large-4, ...): a stock RoPE host has ONE inv_freq "
                    "table for every layer, so it cannot rotate one layer and not the next (truncating frequencies "
                    "is measured not payable: partial-rope-host, SmolLM2 83% identical); a mixed host's NoPE layers "
                    "cannot play rotating layers (layer alignment: rope_fits does not model which host layers "
                    "read which table). The rot gate prices the rest: the NoPE layers' zero-frequency slots "
                    "(rope_sched negative entries) must fit a host table beside the rotating ones"),
    # round 2, pass 3, a design decision: DSA is dense attention. The indexer keeps the index_topk best-scoring keys
    # per query (DeepSeek-V3.2, GLM-5, Mistral-Large-4, MiniMax-M3 / MoBA / InfLLM-v2 block top-k); while the context
    # is <= index_topk (2048 tokens for V3.2 / GLM-5) every key is kept and the layer IS dense attention, exactly.
    # Beyond that it is a content-dependent mask -- the context caveat: the headline relation does not price
    # context (USE_CONTEXT off), and DSA edges hold only up to index_topk (or topk x block) tokens. A DSA host plays
    # dense attention by setting index_topk >= the context (a config value), so the feature is free both ways.
    "dsa":        ("free", "free",
                   "DeepSeek sparse attention (an indexer keeps the index_topk best keys per query): treated as dense "
                   "attention -- exact while the context is <= index_topk; a DSA host sets index_topk >= the context"),
    "kv_compress": ("wall", "wall",
                    "compressed KV pools (DeepSeek-V4 compress_ratios: CSA / HCA softmax-gated pools of 4 / 128 tokens): "
                    "a pooled key is no single token's key at any context length, and a pooling host cannot unpool"),
    # --- round 2, pass 1 (classification): features the configs carry that no rule prices yet. Each is a
    # conservative wall; later passes decide which have a currency.
    "post_ln":    ("wall", "wall",
                   "post-LayerNorm blocks x' = LN(x + f(x)) with no final norm (OPT-350m do_layer_norm_before = false, "
                   "GPT-1): a pre-norm host cannot replace its stream by a normed one, and a post-LN host cannot keep "
                   "an un-normed stream (the universal host's stream norms carry it)"),
    "clip_qkv":   ("wall", "wall",
                   "q / k / v clamped to [-c, c] (OLMo-1 clip_qkv, MPT attn_config.clip_qkv): exact only while no "
                   "projection reaches the clamp (a regime, not measured)"),
    "act_clip":   ("wall", "wall",
                   "a clamped SwiGLU without swigluoai's (up + 1): gate <= L, |up| <= L (DeepSeek-V4 swiglu_limit, "
                   "Step-3.5 swiglu_limits, Hy4, GLM-5-Next): exact only inside the clamp (a regime, not measured)"),
    "router_sqrtsoftplus": ("free", "free",
                            "expert scores sqrt(softplus(l)) (DeepSeek-V4 scoring_func sqrtsoftplus): priced by moe_why "
                            "(router score; V4's ACT2FN[scoring_func] takes sigmoid or sqrtsoftplus from the config)"),
    "hash_route": ("wall", "wall",
                   "hash routing (DeepSeek-V4 num_hash_layers: tid2eid[input_ids] fixes the experts of the first layers, "
                   "the learned gate only weights them): a token-id lookup no learned router forms, and a hash-routed "
                   "layer cannot select by content"),
    "selfcond":   ("wall", "free",
                   "self-conditioning (DiffusionGemma, LLaDA2.1-ORL, QwenDLM-ORL): the previous step's logits fed back "
                   "as an extra input, softmax(prev) E -> MLP -> added; no host without it reads that input; a "
                   "self-conditioning host zeroes it (down_proj = 0 adds exactly 0) where nothing else is on the path"),
    # round 2, pass 4: the self-conditioning path of DiffusionGemma ends in a weight-free RMSNorm of (embedding +
    # signal) that runs on every step, the zero signal included (DiffusionGemmaSelfConditioning.post_norm,
    # with_scale=False; the decoder passes zeros on the first step): zeroing the signal leaves a per-token rescaled
    # embedding, which no residual stream without it reproduces. QwenDLM-ORL sets self_conditioning_post_norm (its
    # modeling_sdar.py is not public: taken as the same norm). LLaDA2.1-ORL has none (self_conditioning_post_norm
    # false), so only the input is a wall there.
    "selfcond_norm": ("wall", "wall",
                      "a self-conditioning path ending in an always-on weight-free RMSNorm of the embeddings "
                      "(DiffusionGemma post_norm, QwenDLM-ORL self_conditioning_post_norm): zeroing the signal leaves "
                      "the per-token rescaled embedding, so the host cannot switch it off"),
    # round 2, pass 4: Zamba / Zamba2's SHARED transformer block (Zamba2HybridLayer / ZambaHybridLayer) reads
    # concat(hidden, original embeddings), 2h wide, RMS-normed jointly, has NO residual inside (attention ->
    # pre_ff_layernorm -> MLP), and its output enters the next Mamba layer through a per-invocation linear; Zamba2
    # adds per-invocation LoRA adapters (gate_up always, q / k / v under use_shared_attention_adapter). No plain layer
    # forms that block and no Zamba layer drops it; the universal host has no Zamba loader. A Zamba2 host plays a
    # Zamba2 target with dims zamba_mem / zamba_rank / zamba_attn_adapt (emul_matrix: _rec_ok).
    "zamba_shared": ("wall", "wall",
                     "Zamba / Zamba2 shared transformer block: concat(x, embedding) input with a joint norm, no inner "
                     "residual, per-invocation linear and LoRA adapters -- no plain layer forms it, and a Zamba "
                     "host's hybrid layers cannot drop it"),
}
# --- round 3, batch 1 (classification): features the configs / code carry that p5 did not read. Each is keyed by its
# spec (verify/r3/specs/<spec>.md). The host side is "free" only where PROOFS.md has the switch-off construction
# (named per entry); the target sides are walls (no theorem needed for a wall).
FEATURES = _R3Dict(FEATURES, add={
    "mm200_core_keys_unread": {
        "out_gate_post": ("wall", "free",
                          "attention output gate AFTER o_proj, o_proj(attn) * sigmoid(W_g x) (MM200 use_gated_attention, "
                          "modeling_mm200.py:229-232): a data-dependent product on the hidden-wide output, not Qwen3-"
                          "Next's pre-o_proj gate (neither plays the other unless o_proj is diagonal); a gated host sets "
                          "W_g = 0 (gate 1/2) and doubles o_proj (Lean out_gate_off, the same identity after o_proj)")},
    "no_model_type_longcat": {
        "scmoe": ("wall", "free",
                  "LongCat's shortcut-connected MoE block (modeling_longcat.py:514-579): per layer two attention + "
                  "dense-MLP sublayers, the MoE reads the first sublayer's output and joins after the SECOND, so attn1 / "
                  "mlp1 never see it -- no sequential pre-norm stack adds a branch after the next attention (OPEN); a "
                  "LongCat host zeroes attn1 / mlp1 / mlp0 and is one sequential MoE layer (Lean zeroed_sublayer)")},
    "text_encoder_input_embeds": {
        "enc_input": ("wall", "free",
                      "the LM's text inputs are a bidirectional encoder's projected output (Breeze-TTS: a T5Gemma2 text "
                      "encoder, breeze.py:947-1004, :1352-1491): no per-token embedding of a causal host forms them "
                      "(OPEN); a Breeze host feeds plain tokens through its own embedding path, the encoder unused (Lean "
                      "extra_input_off / zeroed_sublayer)")},
    "cross_attention_decoder_unflagged": {
        "cross_attn": ("wall", "free",
                       "per-layer cross-attention to an encoder memory (Parler-TTS decoder over the T5 description "
                       "encoder, modeling_parler_tts.py:967, :1036-1052): a decoder-only host has no encoder input "
                       "(OPEN, the extra-input walls' class); a cross-attention host zeroes its encoder_attn out_proj "
                       "(Lean zeroed_sublayer / extra_input_off)")},
    "fake_quant_forward": {
        "fake_quant": ("free", "wall",
                       "weights ternary-quantised inside the forward (BitCPM-CANN *-unquantized LinearQuantizer, "
                       "modeling_minicpm.py:129-162, every projection): as a target a fixed linear map q(W) any host "
                       "stores; as a host it realises only row-ternary {-a, 0, +a} matrices (keyed by repo id: the "
                       "configs are byte-identical to the stock ones)")},
    "gated_prenorm_lowrank": {
        "gated_norm": ("wall", "free",
                       "a low-rank sigmoid gate on the normed sublayer input, y * sigmoid(fc2(silu(fc1 y))) (A.X-K2 "
                       "AXK2GatedRMSNorm, modular_axk2.py:185-209; Grug-MoE GrugMoeGatedNorm rank 128, also after the "
                       "embedding and final norms): a data-dependent product no plain pre-norm forms (OPEN); a gated host "
                       "sets fc2 = 0 (gate 1/2) and doubles the norm weight (Lean out_gate_off, shared_gate_half). "
                       "Between gated points emul_matrix also needs the host's gate rank and sites to cover the "
                       "target's (dims gnorm_rank / gnorm_sites; Lean lowrank_pad)")},
    "multimax_logits": {
        # (host side a wall too: t = 0 is the identity, but PROOFS.md has no multimax_zero; a strictly increasing map
        # would make the target side greedy-exact, needs strictMono_logit_greedy and the checkpoint's r / t)
        "multimax": ("wall", "wall",
                     "an elementwise piecewise-quadratic map on the logits, l + t0 relu(r0 - l) + t1 relu(l - r1) + t2 "
                     "relu(r2 - l)^2 + t3 relu(l - r3)^2 (ERNIE-5.5 multimax lm_head, model.py:84-101), buffers from "
                     "the checkpoint: argmax-preserving only if monotone on the reached logits (unknown here); the "
                     "host's t = 0 switch-off needs theorem multimax_zero")},
    "norm_placement_keys_unread": {
        "post_norm_attn": ("limit", "wall",
                           "a post-norm on the ATTENTION output only (Needle post_attn_norm, IQuest-Q1 attn_out_norm; no "
                           "norm after the MLP): a full-sandwich host plays it with its MLP post-norm switched off by the "
                           "eps lever (Lean postNorm_layer_bound; payable only on a host with eps_lever, emul_matrix), a "
                           "post_norm_attn host has no MLP post-norm to play a full-sandwich target")},
})


# --- round 3, batch 2a: the attention output gate (PLAN decision 2: ONE key) ---------------------------------------
# The FEATURES key is out_gate[_head][_bias][_softplus|_silu] (or out_gate_qlat): granularity (one gate per head
# broadcast over head_dim, W_g: hidden -> heads, vs one per channel), a bias inside the gate, the gate's activation
# (sigmoid unless suffixed). Each is its own feature, a wall on the target side (no ungated layer forms a data-
# dependent product: OPEN in PROOFS.md) --
#   host side: a sigmoid gate host plays an ungated target exactly (Lean out_gate_off: the gate projection 0, sigma 1/2,
#     o_proj doubled -- a biased gate's projection is affine, its bias zeroed with it); a softplus host would need
#     softplus_gate_off (gate ln 2, o_proj / ln 2) and a silu gate cannot (silu(0) = 0): walls;
#   between gates (FEAT_NEST): an elementwise gate with its rows tied per head IS the head-wise gate, but PROOFS.md has no
#     out_gate_head_sub_elem yet, and a biased gate with b = 0 is the bias-free one but has no theorem either (needs
#     theorem out_gate_zero_bias, the qkn_ln_zero_bias analogue): so every pair of distinct gate keys is a wall until they
#     land (FEAT_NEST_PENDING). A head-wise host can never play an elementwise target (rank: needs theorem
#     out_gate_elem_not_head for the record), nor any gate one of another activation (gate_act_ne), nor an x-input gate
#     ERNIE-5.5's gate of the normed q latent (gate_qlat_not_x).
# Granularity / bias by code (out_gate_by_code_unlisted; each file:line in its spec): head -- LFM3 / LFM3-RD gate_proj
# hidden -> heads, Grug-MoE attn_gate (2 sigmoid, the 2 folds into o_proj), dots3-note g_proj; elementwise -- Qwen4-exp
# (q_proj 2x, chunked), Alice-AI, GPT-S3 / -X3, Hy4, Muse-Glimmer, HRM-Text (bias = attention_bias), Dory (a per-channel
# logit scale: a row scaling of W_g), KLM, Needle; elementwise with a bias -- Beam (gate_bias per head); Ethoric attn_gate
# headwise / elementwise, both bias=True.
OUT_GATE_BY_CODE = _R3Dict({}, add={"out_gate_by_code_unlisted": {
    **{t: "head" for t in ("lfm3", "lfm3_rd", "grug_moe", "dots3_note", "dots3_note_text")},
    **{t: "elem" for t in ("qwen4_exp", "qwen4_exp_text", "alice_ai", "gpts3_hybrid", "gptx3", "hy_v4", "muse_glimmer",
                           "muse_glimmer_text", "dory", "klm", "needle")},
    "beam": "elem_bias"}})


def _out_gate_key(d, mt):
    """The FEATURES key of a config's attention output gate (see above), or None (no gate)."""
    on = bool(d.get("attn_output_gate") or d.get("attention_output_gate") or d.get("mla_use_output_gate")
              or d.get("headwise_attn_output_gate") or d.get("use_head_wise_attn_gate") or d.get("gating") is True
              or (d.get("use_output_gate") and mt != "gla") or d.get("attn_use_output_gate")
              or d.get("gate_attn_act_mode") or d.get("attention_gate_func") or d.get("use_gqa_gate")
              or d.get("gated_attention")
              or d.get("gated_attention_proj_granularity_type")              # Ling-3 (bailing_hybrid v3): head-wise
              or mt in OUT_GATE_TYPES)
    head, bias, act = False, False, ""
    ag = str(d.get("attn_gate") or "").lower()
    if _r3("out_gate_by_code_unlisted"):
        if mt == "ethoric":                                # attention.py:168-173, :257-263
            if ag in ("headwise", "elementwise"):
                on, head, bias = True, ag == "headwise", True
        elif mt == "hrm_text":                             # modeling_hrm_text.py:187-191 (bias = attention_bias)
            on, bias = True, d.get("attention_bias") is True
        elif mt in OUT_GATE_BY_CODE:
            k = OUT_GATE_BY_CODE[mt]
            on, head, bias = True, k.startswith("head"), k.endswith("_bias")
    if _r3("out_gate_headwise"):
        # round 3 (out_gate_headwise): the head-wise forms by config. Laguna / Step-3.7 gating True or "per-head"
        # (modular_laguna.py:289-291; absent = True, False = no gate), Youtu mla_output_gate_type (default head_wise,
        # modeling_youtu_vl_moe.py:884-891), BailingMoeV3 gated_attention_proj_granularity_type head_wise (Ming / Edge0 /
        # Ling-3), Spark / Motif headwise_attn_output_gate, Step-3.5 use_head_wise_attn_gate, Ethoric / slm-tavenel
        # attn_gate (default headwise, hf_model.py:63-66); Motif's elementwise_attn_output_gate is a gate too
        g = d.get("gating", True)
        if mt in ("laguna", "step3p7"):
            on = on and g is not False
            head = g is True or str(g).lower() == "per-head"
        elif mt == "youtu_vl_moe":
            head = str(d.get("mla_output_gate_type", "head_wise")).lower() == "head_wise"
        elif str(d.get("gated_attention_proj_granularity_type") or "").lower() == "head_wise":
            head = True
        elif mt in ("spark2_5", "motif") and d.get("headwise_attn_output_gate") is True:
            head = True
        elif mt == "step3p5" and d.get("use_head_wise_attn_gate") is True:
            head = True
        elif mt in ("ethoric", "slm_tavenel"):
            ag = ag or "headwise"
            if ag in ("headwise", "elementwise"):
                on, head = True, ag == "headwise"
        if mt == "motif" and d.get("elementwise_attn_output_gate"):
            on = True
    if _r3("out_gate_activation"):
        # round 3 (out_gate_activation): Laguna's gate is softplus(g_proj x) (modular_laguna.py:335; gating False: none),
        # K2-Horizon's attention_gate_func softplus (softplus(z, beta = ln 2) = softplus(ln2 z) / ln 2: the same class,
        # the beta folds into W_g / o_proj) or silu (modeling_k2_horizon.py:312-321), Spark's gate_attn_act_mode silu
        if mt == "laguna":
            on = d.get("gating", True) is not False
            act = "softplus" if on else ""
        elif mt == "k2_horizon":
            fn = str(d.get("attention_gate_func") or "").lower()
            act = fn if fn in ("softplus", "silu") else ""
        elif mt == "spark2_5" and d.get("headwise_attn_output_gate") is True:
            act = "silu" if str(d.get("gate_attn_act_mode") or "").lower() == "silu" else ""
    if _r3("out_gate_key_unread") and mt == "ernie5_5_moe" and d.get("use_gated_attn", True) is not False:
        # round 3 (out_gate_key_unread): ERNIE-5.5 gates o by sigmoid(o_gate(q_c)), q_c the normed q latent (vLLM-ERNIE
        # attention.py:823-830, :887, :904; its HCA layers likewise, :187-194, :235)
        return "out_gate_qlat"
    if not on:
        return None
    return "out_gate" + ("_head" if head else "") + ("_bias" if bias else "") + (f"_{act}" if act else "")


# --- round 3, batch 2a: qk-norm and output gates -------------------------------------------------------------------
# The output-gate keys (_out_gate_key above; every combination registered under each of the three gate specs, so no
# key a live spec can produce falls to feats_ok's ("free", "free") default)
def _og_feats():
    out = {}
    for head in ("", "_head"):
        for bias in ("", "_bias"):
            for act in ("", "_softplus", "_silu"):
                k = "out_gate" + head + bias + act
                if k == "out_gate":
                    continue
                gran = "one gate per head (W_g: hidden -> heads)" if head else "one gate per channel"
                if act:
                    host, why = "wall", ("a softplus host would need softplus_gate_off (W_g = 0: gate ln 2, o_proj / ln 2)"
                                         if act == "_softplus" else "silu(0) = 0: no setting makes the gate a constant")
                else:
                    host, why = "free", "the gate projection (and its bias) 0: sigma 1/2, o_proj doubled (Lean out_gate_off)"
                out[k] = ("wall", host,
                          f"attention output gate o * {act[1:] or 'sigmoid'}(W_g x{' + b' if bias else ''}), {gran} "
                          f"(round 3: out_gate_by_code_unlisted / out_gate_headwise / out_gate_activation): a data-"
                          f"dependent product no ungated layer forms, walled from every other gate key (FEAT_NEST); "
                          f"host side: {why}")
    return out


_OG_FEATS = _og_feats()
FEATURES = _R3Dict(FEATURES, add={
    "out_gate_by_code_unlisted": dict(_OG_FEATS),
    "out_gate_headwise": dict(_OG_FEATS),
    "out_gate_activation": dict(_OG_FEATS),
    "out_gate_key_unread": {
        "out_gate_qlat": ("wall", "free",
                          "ERNIE-5.5's output gate o * sigmoid(W_g q_c), q_c the normed q latent (vLLM-ERNIE attention.py:"
                          "887, :904): an x-input gate cannot form a gate of RMSNorm(A x) (needs theorem gate_qlat_not_x); "
                          "the host zeroes o_gate (sigma 1/2) and doubles wo_b (Lean out_gate_off)")},
    "attn_output_head_norm": {
        # (host side: the eps lever through o_proj, o_norm(o / S) S sqrt(eps) -> o, is unmeasured and needs theorem
        # oNorm_lever: a wall until then, and onorm stays out of EPS_LEVER_FEATS)
        "onorm": ("wall", "wall",
                  "a weight-free RMSNorm on each head's attention output before the gate and o_proj (Muse-Spark o_norm, "
                  "modular_muse_spark.py:397-400, :445-447): with one visible key it is a v-norm (Lean vnorm_not_plain), "
                  "so no plain head forms it; the host's switch-off by the eps lever needs theorem oNorm_lever")},
})
# Muse-Spark's o_norm (attn_output_head_norm; BitNet's weighted full-width attn_sub_norm would be its own key, none in
# the snapshot)
ONORM_TYPES = _R3Set((), add={"attn_output_head_norm": {"muse_spark", "muse_spark_text"}})


# --- round 3, batch 2b: positions ----------------------------------------------------------------------------------
FEATURES = _R3Dict(FEATURES, add={
    "position_query_temperature": {
        # host side: scale_h = 0, bias_h = 1 are weights (checkpoint parameters) and make the factor 1 at every position:
        # the host's attention is the plain one (catalog feature:q_temp:host, verify/spot/catalog/r3.py)
        "q_temp": ("wall", "free",
                   "a LEARNED per-head query temperature q * (s_h log(pos + 1) + b_h) on every layer (Mistral-Large-4 "
                   "MistralLarge4TextSoftmaxScaling, tfmain modular_mistral_large4.py:296-309), ungated: log(pos + 1) "
                   "folds into no position-free W_q (needs theorem learned_logpos_not_plain for the wall's record); a "
                   "host sets s_h = 0, b_h = 1"),
        "q_temp_nope": ("wall", "wall",
                        "a code-fixed query temperature (1 + (pos + 1) / 2048)^0.1 on the NoPE full layers (Beam "
                        "modular_beam.py:172-174, :207-208, :420-428; 1.0005 at position 0): no constant folds it, and a "
                        "host cannot switch its constants off")},
    "fope_positions": {
        # host side: C_h = S_h = I would leave plain RoPE on the KEPT frequencies only (inv_freq > 2 pi / mpe, the
        # dropped pairs a fixed (I + R) map), so a target frequency below the host's cut cannot be placed: a wall until a
        # rule checks the frequency range (needs theorem fope_host_off)
        "fope": ("wall", "wall",
                 "Fourier position embedding (Intern-S1-Pro rope_scaling fope_init_factor / fope_sep_heads / "
                 "num_inv_freq: modeling_rope_utils.py:134-141, :583-600; modeling_interns1_pro.py:866-897, :934): "
                 "per-KV-head learned cos / sin mixing (cos(w t) C_h, sin(w t) S_h) is no rotation and its logit no "
                 "function of i - j (needs theorem fope_not_rope); a host's switch-off keeps only the frequencies above "
                 "2 pi / max_position_embeddings (needs theorem fope_host_off)")},
    "content_warped_rope": {
        # host side: the metric projections W = 0 (weights) give mass_delta = spin_delta = 0, phase = t inv_freq: the
        # plain table whatever the clock state (catalog feature:rope_warp:host)
        "rope_warp": ("wall", "free",
                      "a content-warped RoPE phase t w (1 + m r(t)) + s (t / (T - 1))^2 whose deltas m, s = c tanh(W z) "
                      "read the masked mean of the WHOLE sequence's hidden states (GPT-X2 use_symplectic_rope, "
                      "modeling_gptx2.py:706-744, :96, :869): non-causal (position t reads t' > t, as "
                      "causal_ne_bidirectional), so no causal host forms it; a host zeroes W")},
})


# fr-positions (fp_nope_layer_host): the host-side features a RoPE + NoPE host drops for a target without NoPE layers --
# its NoPE layers are zeroed (identity blocks) and the target runs on its RoPE layers (emul_matrix's rotating-layer
# Hall), so neither the NoPE layers nor Beam's temperature on them touch the target (Lean nope_mix_host_skip)
NOPE_HOST_OFF = frozenset({"nope_layers", "q_temp_nope"})


# --- round 3, batch 2c: windows and masks ----------------------------------------------------------------------------
# (blocksparse_mask: a design decision -- Phi-3-small's block-sparse layers are DENSE attention, as DSA. Its
# BlockSparseParams mask (triton_flash_blocksparse_attn.py:152-180: block-causal, num_local_blocks x block_size = 1024
# local tokens plus every vert_stride-th key block, per-head shifted) keeps every key while the context is <= 1024 and
# the layer IS dense causal attention there, exactly; beyond that it is a fixed sparse pattern -- the context caveat,
# as DSA's: the headline does not price context, and the phi3small edges hold only up to 1024 tokens. No layer kind
# "PU", no feature: a no-op in the relation.)
FEATURES = _R3Dict(FEATURES, add={
    "swa_prefill_key_truncation": {
        # (exact only for contexts <= sliding_window_size: a context caveat like DSA's, not taken)
        "swa_trunc": ("wall", "wall",
                      "Meditor-1 / -2 sliding layers truncate K / V to the last sliding_window_size keys BEFORE a "
                      "top-left-aligned causal SDPA (modeling_meditor2.py:171-188, modeling_meditor.py:164-185; the band "
                      "branch is dead): a prefill longer than the window attends future tokens, which no causal stack "
                      "forms (causal_stack); its leaky sliding layers play neither full nor band layers (needs theorem: "
                      "no causal stack equals top-left causal attention over the last W keys at L > W)")},
    "doge_dynamic_mask": {
        # host side: A = 0 (a weight) makes the bias exp(0) = 1 on every unmasked key, a common logit shift
        # (Lean renorm_softmax_shift); keep_window_size >= T (a window size transformers' DogeAttention reads) keeps the
        # top-k off (catalog feature:dyn_mask:host)
        "dyn_mask": ("wall", "free",
                     "Doge's dynamic mask: a per-key additive logit bias exp(A softplus(dt_proj(v_j))) (+ only the top "
                     "keep_window_size keys) (tf5 modeling_doge.py:303-371; the remote Doge-320M per query head): a "
                     "content-dependent bias no plain layer forms (needs theorem: a value-dependent per-key logit bias "
                     "is not plain pre-norm attention; const_readout_channel the pre-norm half); a Doge host sets A = 0 "
                     "and keep_window_size >= T")},
    "eos_document_mask": {
        # host side: the emulator picks the host's token ids (vocab ignored), so the host's eos id never occurs and its
        # document mask is the causal mask (restriction; catalog feature:doc_mask:host)
        "doc_mask": ("wall", "free",
                     "in-sequence documents split at the eos token: same-document attention and conv / recurrent / "
                     "n-gram resets (Ethoric doc_masking, modeling_ethoric.py:435-443 / attention.py:80-127; KLM "
                     "segments, modeling_klm.py:66-84, :327-332): a token-id-dependent hard mask no plain layer forms "
                     "(needs theorem: no plain causal stack equals an eos-segmented one on inputs containing eos); a "
                     "doc-mask host never sees its eos id")},
    "attn_output_saturation": {
        # (host side: the thresholds are cap values the remote config sets -- free in principle, but remote code gets no
        # lever; d_ros_config.enabled is a feature switch)
        "attn_out_sat": ("wall", "wall",
                         "attention-output saturation (Qwen3-D-ROS d_ros_config, modeling_qwen3_eq.py:85-166): per-band "
                         "threshold / tanh-knee compression of the attention output on the middle layers, dry / wet "
                         "mixed: the identity only below the thresholds (a regime, not measured; needs theorem: a "
                         "thresholded saturation is the identity only below its threshold)")},
})


# --- round 3, batch 2d: heads and MLA ---------------------------------------------------------------------------------
# Every target side is a wall (no theorem needed); a host side is "free" only with a catalog instance built from the
# target's weights (verify/spot/catalog/r3.py: feature:<key>:host) or the PROOFS.md lemma named per entry.
FEATURES = _R3Dict(FEATURES, add={
    "vha_head_mix": {
        # host side: U = V = 0 gives m = I (catalog feature:vha:host; no PROOFS.md lemma, needs theorem vha_off for the
        # record); target side OPEN (a fixed head mix behind a data-dependent gate / a grouped wo_a)
        "vha": ("wall", "free",
                "ERNIE-5.5's VHAPostMix, o[h] <- sum_g (I + V U^T)[h, g] o[g] across the heads (vLLM-ERNIE attention.py:"
                "85, :110-146, used at :229 / :903-904 before the grouped wo_a / the sigmoid gate): it folds into no "
                "o_proj behind those; a host sets U = V = 0")},
    "diff_attn": {
        "diff_attn": ("wall", "wall",
                      "Motif-3 differential attention a_s - sigmoid(lambda(x)) a_n, the noise head on its signals' K / V "
                      "(modeling_motif.py:555-557, :619, :687, :754-766): a data-weighted second head no plain head forms "
                      "(needs theorem diff_attn_ne_plain), and sigmoid > 0 never drops it on a host")},
    "xsa_self_value_projection": {
        "xsa": ("wall", "wall",
                "exclusive self attention y - (y . v_i / |v_i|^2) v_i with the token's OWN value (GPT-S / GPT-X2 "
                "xsa_projection, modeling_gptx2.py:106-111; Grug-MoE always, skyrl_train_models_grug_moe.py:743-745): a "
                "per-token rank-1 removal no linear map / gate forms (needs theorem xsa_ne_plain), and no weight makes "
                "it the identity on a host"),
        # host side: xsa_alpha = 0 (tanh 0 = 0) is the plain head (catalog feature:xsa_scaled:host). A unit-XSA target on
        # a scaled host (tanh(alpha) -> 1) is FEAT_NEST_PENDING: needs theorem xsa_alpha_limit
        "xsa_scaled": ("wall", "free",
                       "Limite's scaled XSA y - tanh(alpha_h) (y . v_hat) v_hat on xsa_layers (modeling_limite.py:187, "
                       ":207-208, :367-395): a host sets alpha_h = 0")},
    "canon_conv_layers": {
        # host side: zero conv weights (residual) / identity taps give the plain block (catalog feature:canon:host); a
        # Canon target needs a Canon host with the same positions / kernel (_r3_masks "canon cover"; a covering host needs
        # theorem canon_pad)
        "canon": ("wall", "free",
                  "Canon layers: residual causal depthwise short convs over the last k tokens (dhara_ar canon_set, "
                  "modeling_dhara_ar.py:46-100; Ethoric canon, modeling_ethoric.py:38-74; Needle qkv_conv_taps, cactus "
                  "architecture.py:184-203): token mixing inside the block no plain layer forms; a host zeroes the convs")},
    "cca_conv_qk_attention": {
        "cca": ("wall", "wall",
                "ZAYA1's CCA (modeling_zaya1_vl.py:676-888, built unconditionally :943-970): q / k from a causal conv "
                "over a compressed latent, half of every value from x_{t-1}, an eps-free L2 norm of q / k with a learned "
                "temperature: token mixing in the projections (needs theorem cca_ne_plain), and the host's norm bounds "
                "every logit (the qk-norm fold, not payable)")},
    "moa_routed_attention_heads": {
        # host side: tied experts + renormalised gates give one block's attention (Lean tied_moe_is_dense; catalog
        # feature:moa:host), its output bias 0; the host offers n_kv q heads (dims_of). MoA <- MoA: equal (E, k)
        "moa": ("wall", "free",
                "JetMoE's mixture of attention heads: top-k routed (q_proj, o_proj) experts over one shared KV "
                "(modular_jetmoe.py:106-149, :212-373): heads selected per token, discontinuous at the routing boundary "
                "(continuous_host_ne_top1_moe); a host ties its experts")},
    "memory_attention": {
        # host side: the memory's output projection 0 (catalog feature:mem_attn:host; needs theorem mem_off for the
        # record); target side OPEN as kv_compress (pool_one)
        "mem_attn": ("wall", "free",
                     "a summary / top-capacity memory beside the attention: learned slot queries pool each chunk and "
                     "tokens attend to earlier chunks' summaries (haru CausalSummaryMemory model.py:315-445; Bluebird "
                     "fabric_runtime.py:209-296), or a content-selected top-capacity key memory (EdgeRock "
                     "modeling_edgerock.py:122-322): pooled / selected keys no plain layer forms; a host zeroes read_o")},
    "mova_routed_values": {
        "mova": ("wall", "wall",
                 "K2-Horizon / Jais-3 MoVA routed value experts V = s sum_k w_k silu(W_k x) (modeling_k2_horizon.py:351-"
                 "447): V not linear in x (needs theorem mova_not_linear_v); the host's linear limit is unmeasured")},
    "mla_query_raw_latent_skip": {
        # host side: the raw-latent columns of q_gate_proj 0 (catalog feature:mla_q_raw:host); a q_latent = 0 MLA target
        # stays walled by the MLA latent rule
        "mla_q_raw": ("wall", "free",
                      "A.X-K2's MLA query q_gate_proj(cat(RMSNorm(q_a x), q_a x)) (modular_axk2.py:285-291, :327-331): a "
                      "plain MLA query reads only the normed latent (needs theorem: the ray argument of mla_value_ray on "
                      "the query); a host zeroes the raw half")},
    "hybrid_mla_layers_unread": {
        "mla_qmix": ("wall", "wall",
                     "Sabi's shared pre-RoPE MLA: per-head q_norm, then learned rotary_query / residual_query mixes of the "
                     "normed rope half, no latent norm, no k_norm (modeling_sabi_mla.py:88-148): no q-norm-then-RoPE head "
                     "forms the twice-read rope half (needs theorem), and the host cannot drop it")},
    "kv_compress_variant": {
        "kv_compress_v41": ("wall", "wall",
                            "DeepSeek-V4.1 compressed KV: non-overlapping ratio-2 softmax pools without a position bias, "
                            "ratio-1 per-token norm(wkv x) keys, pools built on kv_source_layers and read by later layers "
                            "(inference/model.py:429-485, :619-660, :740-764) -- not V4's CSA / HCA pools (kv_compress): "
                            "walled from them both ways (needs theorem: the two pool kinds differ on some input)")},
    "b_clip_qkv_host": {
        # host config lever: clip_qkv None skips the clamp (tf5 modeling_olmo.py:245-248, modeling_olmoe.py:253-256,
        # modeling_mpt.py:97-98) -- a cap value, shape-preserving (Lean clampSym_id; catalog feature:clip_qkv:host)
        "clip_qkv": ("wall", "free",
                     "q / k / v clamped to [-c, c] (OLMo-1 clip_qkv, MPT attn_config.clip_qkv): exact only while no "
                     "projection reaches the clamp (a regime, not measured); a clamped host sets clip_qkv None (off)")},
})


# --- round 3, batch 3a: MoE and router --------------------------------------------------------------------------------
# The MoE-block forms the routing rules compare are MOE_SIG fields (moe_why); these are the router / block features.
# Target sides are walls; a host side is "free" only with its PROOFS.md lemma or a catalog instance (verify/spot/catalog/
# r3.py feature:<key>:host).
FEATURES = _R3Dict(FEATURES, add={
    "router_bias_unread": {
        # host side: b = 0 (Lean router_bias_off; catalog feature:router_bias_logit:host). Cross-domain (score vs logit)
        # both ways a wall: needs theorem router_bias_domain_ne
        "router_bias_logit": ("wall", "free",
                              "a LOGIT-domain selection-only bias: top-k on l + b, weights from the unbiased score "
                              "(Grug-MoE skyrl_train_models_grug_moe.py:544-553, Kolibri-1 modular_kolibri1.py:136-137, "
                              "Youtu-A7B quantile balancing modeling_youtu_vl_moe.py:518-524): for a sigmoid router no "
                              "score-domain bias selects the same experts unless b is constant, and a bias-free router "
                              "selects on its weighting logits (router_bias_not_bias_free); a host sets b = 0")},
    "mlp_router": {
        "router_mlp": ("wall", "wall",
                       "a tanh-MLP router W2 tanh(W1 x) (LLaMA-MoE gate_network mlp, modeling_llama_moe_hf.py:440-455, "
                       ":484-497): no linear router forms it (needs theorem tanh_router_ne_linear); the host's linear "
                       "limit tanh(eps W x) / eps -> W x is unmeasured (needs theorem tanh_router_limit)")},
    "moe_capacity_drop": {
        "moe_capacity": ("wall", "wall",
                         "per-forward expert capacity with dropped assignments (Hunyuan-A13B remote topkgating, "
                         "modeling_hunyuan.py:74-140: capacity max(k, k S / E) over the batch's S tokens): a token's "
                         "experts depend on the other tokens of the forward (needs theorem capacity_drop_not_tokenwise); "
                         "moe_drop_tokens is not read, so a host cannot go dropless")},
    "block_routing": {
        # host config lever: expert_capacity = num_experts allows every expert per block, so the gate is the plain
        # top-k on score + bias (Lean topk_all; catalog feature:block_routing:host). Block <- block: moe_why (the same
        # block routing at m = 1; copies need theorem block_routing_copies)
        "block_routing": ("wall", "free",
                          "LLaDA2.2 block routing (modeling_llada2_moe.py:244-296): per block of block_size tokens only "
                          "the expert_capacity experts with the best block-max score are allowed: a token's experts "
                          "depend on its block (needs theorem block_routing_not_tokenwise); a host sets expert_capacity "
                          "= num_experts")},
    "expert_output_norm": {
        # host side: the eps lever on each expert's output (epsLever_eq with the gain moved to post_expert_proj) is a
        # limit not plumbed into the lever slice here: a wall (conservative)
        "expert_norm": ("wall", "wall",
                        "an RMSNorm on every routed expert's output before its routing weight (Muse-Spark "
                        "post_expert_norm, modular_muse_spark.py:340-371): each expert contributes a unit-RMS vector, "
                        "which no plain expert bank forms (needs theorem expert_norm_not_plain)")},
    "gemma4_branch_post_norms": {
        # the record of MOE_SIG branch (moe_why prices it): the host side needs theorem double_rmsnorm_fold
        "moe_branch_norm": ("wall", "wall",
                            "Gemma-4 / DiffusionGemma MoE layers: N_post(N_post1(mlp(.)) + N_post2(experts(.))) "
                            "(modeling_gemma4.py:1410-1429): two separately normed branches under one more norm; a "
                            "host runs one branch through two norms (the fold needs theorem double_rmsnorm_fold)")},
    "arctic_dense_residual_parallel_moe": {
        # host side: the MoE zeroed (expert down = 0), the residual MLP plays a sequential dense FFN at 1x (Lean
        # zeroed_sublayer; catalog feature:par_moe_res:host); MOE_SIG par_res keeps MoE targets off it (the 2x-depth
        # pairing, seq_sub_parallel, is not plumbed) and caps a dense target's width at hidden_size
        "par_moe_res": ("wall", "free",
                        "Arctic parallel_attn_mlp_res (modeling_arctic.py:1042-1050, :1083-1110): h = x + attn(n1 x); "
                        "out = h + residual_mlp(n_r h) + moe(n2 x) -- the MoE reads the layer input beside attention "
                        "(the parallel <- sequential carry is refuted under a pre-norm: parallel_sub_seq_prenorm_fails)")},
})


# --- round 3, batch 3b: FFN widths and activation clamps ----------------------------------------------------------------
# The families whose code reads the SwiGLU clamp limit from the config (b_act_clip_host): tf5 DeepSeek-V4 / MemT /
# GLM-5-Next / Hy4 (gate.clamp(max=L), up.clamp(-L, L); L = inf is the identity on every finite input), the V4.1
# reference and Step-3.5 / 3.7 (a limit 0 / None builds no clamp); Ling-3's per-layer lists (act_clip_per_layer_list:
# `limit_list[layer_idx] or None`). Not exaone_moe: tf5 ExaoneMoe never reads swiglu_limits (PLAN decision 5).
ACT_CLIP_CFG_TYPES = _R3Set({"deepseek_v4", "memt_deepseek_v4", "deepseek_v41", "deepseek_v41_text", "glm5_next",
                             "glm5_next_text", "hy_v4", "step3p5", "step3p7"},
                            add={"act_clip_per_layer_list": {"bailing_hybrid"}})
# act_clip_order (a non-issue, recorded): the clamp before SiLU (DeepSeek-V4, GLM-5-Next, Hy4, MemT) and after it
# (Step-3.5 / 3.7, Ling-3) convert exactly through the free limit L' = silu^-1(L) / silu(L) and an up / down rescale
# (needs theorem act_clip_order_swap; ffn_smul), so act_clip <- act_clip is credited regardless of order. The residual
# the feature does not compare is the per-layer / per-branch limit PATTERN (Step-3.5 clamps only its last layers, at 7
# routed / 16 shared; a uniform-limit V4 host cannot leave the others unclamped outside act_clip_regime's bounded
# context): a follow-up (dims act_clip_sched), not this batch.
# The Dory / BitNet intermediate norm (ffn_inner_norm): down(N(act(g) u)), N the weight-free L2 norm (Dory, eps 1e-6 in
# code) or a weighted RMSNorm (BitNet ffn_sub_norm) over the intermediate width
FFN_NORM_TYPES = _R3Set(set(), add={"ffn_inner_norm": {"dory", "bitnet"}})
FEATURES = _R3Dict(FEATURES, add={
    "b_act_clip_host": {
        # host config lever: swiglu_limit = inf (tf5 V4 / GLM-5-Next / Hy4: clamp(x, -inf, inf) = x for every finite x)
        # or 0 / None (V4.1, Step-3.5 / 3.7, Ling-3: no clamp is built) -- a cap value, shape-preserving (Lean
        # act_clip_regime: exact once L is above every activation, which inf / off is on every input; catalog
        # feature:act_clip:host). The target side stays a wall (analytic_ne_clamp, act_clip_unclamped_fails)
        "act_clip": ("wall", "free",
                     "a clamped SwiGLU without swigluoai's (up + 1): gate <= L, |up| <= L (DeepSeek-V4 swiglu_limit, "
                     "Step-3.5 swiglu_limits, Ling-3 per-layer lists, Hy4, GLM-5-Next): no analytic host forms the clamp "
                     "(act_clip_unclamped_fails); a clamped host sets its limit to inf / 0 = off")},
    "ffn_inner_norm": {
        # target: needs theorem ffn_norm_not_plain (at small inputs a plain FFN's output -> 0, the normed one keeps a
        # unit-norm direction); host: the eps lever through down_proj (up rows / S, down x S sqrt(eps): epsLever_eq) is
        # a limit, unmeasured here: a wall both ways (the 4 Dory points are walled by ngpt anyway)
        "ffn_norm": ("wall", "wall",
                     "a norm on the FFN's intermediate activation before down_proj (Dory ffn_sub_norm = weight-free L2, "
                     "modeling_dory.py:431-436; BitNet's weighted ffn_sub_norm, modeling_bitnet.py:74-77): the FFN "
                     "output no longer scales with its input, which no plain FFN forms")},
})
# --- round 3, batch 4a: loops ------------------------------------------------------------------------------------------
# looped_sub_untied (PROOFS.md "looped ⊆ untied": K loops of a b-layer block = a Kb-layer untied stack with tied weights)
# is the only loop rule; its premise fails where the loop boundary does more than hand the stream on. Each such
# boundary is a FEATURES wall (a target needs a host doing the same at the same boundaries), the host side free only by
# a host config lever or a weight setting with a construction.
FEATURES = _R3Dict(FEATURES, add={
    "loop_boundary_norm": {
        # target: needs theorem looped_bnorm_not_unrolled (an untied pre-norm stack x -> x + f(N x) has no layer whose
        # output stream is N(x), cf. the post_ln walls). Host: Ouro / HRM / Dory hard-code it (a scale-invariant renorm
        # no weight switches off): a wall
        "loop_bnorm": ("wall", "wall",
                       "the residual stream renormalised at every loop / stack-pass boundary, fixed in code (Ouro "
                       "self.norm after every ut step, modeling_ouro.py:654; HRM-Text's weight-free stack final_norm, "
                       "modular_hrm_text.py:225-250; Dory h = L2(h) + e at every loop start, modeling_dory.py:619-620)"),
        # host config lever: skip_loop_final_norm = true (Nanbeige4.2 modeling_nanbeige.py:2401-2412, qwen3loop
        # :262-263): a norm on/off flag, shape-preserving -- the stream is normed once at the end, the plain looped
        # stack (catalog feature:loop_bnorm_cfg:host). The target side is the same wall as loop_bnorm
        "loop_bnorm_cfg": ("wall", "free",
                           "the stream renormalised by the shared final norm after every loop pass unless "
                           "skip_loop_final_norm (Nanbeige4.2 / qwen3loop num_loops > 1): an untied pre-norm stack "
                           "has no mid-stack renorm; a host sets the flag")},
    "loop_input_injection": {
        # needs theorem loop_inject_not_prenorm (e is added to the stream: a magnitude consumer, so the vanishing
        # carry's limit does not cover it); a wall both ways (no switch: Dory's add runs at every loop start)
        "loop_inject": ("wall", "wall",
                        "the loop input re-derived from the prelude output / embeddings at every iteration (Dory h = "
                        "L2(h) + e, modeling_dory.py:618-628; Huginn adapter(cat[x, e]); LFM3-RD inject_feedback past "
                        "num_motifs loops): a second stream an unrolled pre-norm stack does not carry")},
    "loop_kv_share_gated_mix": {
        # needs theorem gated_two_softmax_mix_not_single; a wall both ways (a window >= context removes the local
        # window, not the gated mix with loop 1's K / V)
        "loop_kv_mix": ("wall", "wall",
                        "loops >= 2 mix attention over loop 1's K / V of the same layer with a local-window attention "
                        "over the current loop, by a per-head sigmoid gate of q (IQuest-Loop forward_loop2, "
                        "modeling_iquestloopcoder.py:554-700, LoopGateProjection :433-451)")},
    "hrm_hierarchical_loop": {
        # needs theorem two_state_loop_not_single_stream; host: with the L-stack zero z_L = final_norm(z_H) (a
        # weight-free norm of a nonzero input), not 0 -- a wall both ways
        "hrm_two_state": ("wall", "wall",
                          "two residual states z_L, z_H summed into each other's stack input every pass (HRM-Text "
                          "modular_hrm_text.py:348-400: z_L = L(z_L + z_H) L_cycles times, then z_H = H(z_H + z_L))")},
    "per_token_early_exit": {
        # target: a per-token, data-dependent readout depth (no fixed-depth host picks it; a limit candidate after a
        # measurement of the early-exit fraction). Host: the gate weight 0 / bias -100 would make every token read the
        # last loop, but no theorem / measurement records it (needs theorem early_exit_gate_off): a wall both ways for
        # now (Ouro hosts are walled by loop_bnorm anyway)
        "early_exit": ("wall", "wall",
                       "per-token exit at a data-dependent loop step (Ouro early_exit_threshold: the first step whose "
                       "cumulative exit mass reaches the threshold, modeling_ouro.py:770-850; LFM3-RD auto halting)")},
    "lfm3rd_recurrent_controller": {
        # target: needs theorem scaled_core_sum_not_prenorm (p + s * sum(updates) before the coda; folding s into the
        # core's outputs changes what the later core pre-norms read; exact iff s = 1, a weight value). Host: the step
        # scale set to 1 (a weight) at the config's manual T = num_motifs (feedback / refinement inactive) is the plain
        # stack, p + 1 (h - p) = h (scalar_fold at s = 1; catalog feature:rd_controller:host)
        "rd_controller": ("wall", "free",
                          "LFM3-RD: the core's summed update rescaled by a learned step_scale before the coda "
                          "(modeling_lfm3_rd.py:1179-1180), plus feedback injection / gated refinement past num_motifs "
                          "loops")},
})


# --- round 3, batch 4b: residual streams and extra inputs --------------------------------------------------------------
# Hyper-connection VARIANTS (hyper_conn_variant; the qk-norm-variant pattern): "hyper_conn" keeps the wall to single-
# stream models, and every hyper-connection point also carries the "hc_<variant>" key of its code, walled both ways from
# every other variant -- two stream-mixing codes (Sinkhorn comb vs comb^T at finite iterations, the identity carry, the
# gated residual, the lagged pre, a learned vs a mean collapse) are no weight setting of each other (needs theorem: the
# spec's "hyper-connection variants" wall, OPEN in PROOFS.md "hyper_conn, attn_res (both)"). hc_sinkhorn_iters / hc_eps /
# hc_magnitude are config values inside a variant. By code (verify/r3/specs/hyper_conn_variant.md):
#   dsv4       DeepSeek-V4 / MemT / ERNIE-5.5: comb^T, Sinkhorn, a learned sigmoid HyperHead collapse
#   glm5_next  GLM-5-Next: V4's HyperConnection, the collapse an unweighted mean of the streams
#   dsv4_nano  nanowhale's remote modeling_deepseek_v4.py: comb (not comb^T), its own Sinkhorn
#   dsv41      DeepSeek-V4.1: the attention reads the PREVIOUS FFN's pre (a lagged pre), the last FFN's pre collapses
#   xing       Xing4.0: clamped comb logits, pre without hc_eps, comb (not comb^T), mean collapse
#   hy4        Hy4: pre / post only, the identity carry (no comb)
#   qwen4_exp  Qwen4-exp GatedResidual: grouped RMSNorm, low-rank mix, mean over streams, no Sinkhorn
#   motif / needle / san  Motif-3's MHCLayer; Cactus needle's fixed lane offsets, 20 log-domain Sinkhorn iterations;
#              SAN, its port with 14
# The one candidate nesting, a glm5_next target on a dsv4 host (V4's learned HyperHead with hc_fn = 0, base =
# logit(1 / hc_mult - hc_eps) is the mean), has no PROOFS.md theorem or measurement: FEAT_NEST_PENDING (hc_head_mean).
HC_VARIANT_BY_TYPE = {"deepseek_v4": "dsv4", "memt_deepseek_v4": "dsv4", "ernie5_5_moe": "dsv4",
                      "glm5_next": "glm5_next", "glm5_next_text": "glm5_next",
                      "deepseek_v41": "dsv41", "deepseek_v41_text": "dsv41",
                      "xing4_0": "xing", "hy_v4": "hy4", "qwen4_exp": "qwen4_exp", "qwen4_exp_text": "qwen4_exp",
                      "motif": "motif", "needle": "needle", "san": "san"}
HC_VARIANTS = ("dsv4", "dsv4_nano", "dsv41", "glm5_next", "xing", "hy4", "qwen4_exp", "motif", "needle", "san")


def _hc_variant(d, outer=None):
    """The hyper-connection variant of a hyper_conn config (HC_VARIANT_BY_TYPE; nanowhale's remote code -> dsv4_nano)."""
    mt = str(d.get("model_type") or "").lower()
    if mt not in HC_VARIANT_BY_TYPE:
        return None
    if mt == "deepseek_v4" and "modeling_deepseek_v4" in _auto_map_str(d, outer):
        return "dsv4_nano"
    if mt == "ernie5_5_moe" and d.get("enable_hyper_connections") is False:
        return None
    return HC_VARIANT_BY_TYPE[mt]


def _mhc_lanes(d):
    """hyper_conn_key_unread: the lane count of Cactus needle (extras.mhc_lanes; the cactus TransformerConfig default 4,
    config.py:42) and SAN (top-level mhc_lanes, san_model.py:53, default 4) -- None for other families."""
    mt = str(d.get("model_type") or "").lower()
    if mt == "needle":
        ex = d.get("extras") if isinstance(d.get("extras"), dict) else {}
        return _num(ex["mhc_lanes"]) if "mhc_lanes" in ex else 4
    if mt == "san":
        return _num(d["mhc_lanes"]) if "mhc_lanes" in d else 4
    return None


# round 3 (qwen4_ple_reads_ngrams): Qwen4-exp's "PLE" hashes the previous ngram_size - 1 tokens (XOR of multiplied ids
# mod per-head primes, a signed-sqrt gate per hc stream, a dilated short conv: modular_qwen4_exp.py:643-783) -- an Engram,
# not Gemma's per-token PLE; built only on the layers of a non-empty ple_layer_ids (:796-797)
QWEN4_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})
# round 3 (ngram_memory_unread / qwen4_ple_reads_ngrams): the n-gram memories by code family, each its own hash /
# addressing / injection (wall from every other kind and from DeepSeek-V4.1's "engram"): KLM's exact bigram / trigram
# key banks (modeling_klm.py:94-120; an AttnRes source), Crowd's one-prime bucket of the previous token (modeling_crowd.py:
# 86-97), AllGDN's rolling-hash per (order, head) with a gated conv (modeling_allgdn.py:76-148; the universal host's
# UEngram), needle's XOR-multiply hash with linear dilated taps (cactus architecture.py:95-158, :348-354), Qwen4-exp's
# (above). Host side: the read projection 0 (Lean extra_input_off) -- except KLM, whose zeroed memory is still a source of
# its AttnRes softmax (score q . rms(0) = 0 keeps mass): a soundness wall both ways (moot: KLM is attn_res-walled)
ENGRAM_KINDS = ("engram_qwen4", "engram_klm", "engram_crowd", "engram_allgdn", "engram_needle")


def _engram_kind(d):
    """(the engram_* FEATURES key, its table geometry) of an n-gram memory config, else (None, "")."""
    mt = str(d.get("model_type") or "").lower()
    if _r3("qwen4_ple_reads_ngrams") and mt in QWEN4_TYPES:
        ids = d.get("ple_layer_ids")
        if isinstance(ids, list) and ids:
            g = (tuple(ids), d.get("ngram_size"), d.get("heads_per_ngram"), d.get("ple_embed_dim"),
                 d.get("ple_conv_kernel_size"))
            return "engram_qwen4", repr(g)
        return None, ""
    if not _r3("ngram_memory_unread"):
        return None, ""
    if mt == "klm" and _num(d.get("n_bigrams")) + _num(d.get("n_trigrams")) > 0:
        return "engram_klm", repr((d.get("n_bigrams"), d.get("n_trigrams"), d.get("ngram_dim")))
    if mt == "crowd" and _num(d.get("engram_buckets")) > 0:
        return "engram_crowd", repr((d.get("engram_buckets"),))
    if mt == "allgdn" and isinstance(d.get("engram"), dict) and d["engram"].get("layers"):
        e = d["engram"]
        return "engram_allgdn", repr(tuple((k, repr(e.get(k))) for k in sorted(e)))
    if mt == "needle":
        ex = d.get("extras") if isinstance(d.get("extras"), dict) else {}
        e = ex.get("engram") if isinstance(ex.get("engram"), dict) else {}
        if e.get("sites"):
            return "engram_needle", repr(tuple((k, repr(e.get(k))) for k in sorted(e)))
    return None, ""


# round 3 (value_emb_keys_unread): value embeddings / value residuals by code -- ARLoop / PreNormLoop's one shared table
# gated into V on alternate layers (modeling_arloop.py:37-48, :110-111), sn38-nanochrono's per-layer tables on aux_layers
# (modeling_nanochrono.py:36-48, :124-125; None = every other layer), KLM's value residual v_first mixed by value_mix
# (modeling_klm.py:150-160). The kind (dims ve_kind) is compared for equality on value_emb pairs; the nest ve_layer host
# <- ve_shared target (the shared table copied per layer) needs theorem ve_shared_per_layer, and v_residual / ve_* do not
# nest. The key-path value_emb families (nanochat ve / x0 lambdas / smear, limite ve_gate_channels) are kind "ve"
VALUE_EMB_BY_CODE = _R3Dict({}, add={"value_emb_keys_unread": {"arloop": "ve_shared", "arloop_baseline": "ve_shared",
                                                                "klm": "v_residual"}})


def _ve_kind(d, f):
    """The value-embedding kind of a config whose feature set f has value_emb ("" otherwise)."""
    if not _r3("value_emb_keys_unread") or "value_emb" not in f:
        return ""
    mt = str(d.get("model_type") or "").lower()
    if mt == "sn38-nanochrono":
        return "ve_layer"
    return VALUE_EMB_BY_CODE.get(mt, "ve")


def _ve_by_code(d):
    mt = str(d.get("model_type") or "").lower()
    if not _r3("value_emb_keys_unread"):
        return False
    if mt == "sn38-nanochrono":
        aux = d.get("aux_layers")
        return aux is None or (isinstance(aux, list) and bool(aux))
    return mt in VALUE_EMB_BY_CODE


def _attn_res_sig(d, f):
    """attn_res_keys_unread: the AttnRes source partition "<sublayers per source block>/<window in blocks, 0 = all>"
    of a config whose feature set f has attn_res -- GPT-S3 every sublayer its own source over all ("1/0",
    modeling_gpts3.py:313-371), KLM one source per layer ("2/0", modeling_klm.py:277-352), ARLoop a block every
    attnres_block_size / 2 layers, the embedding and the last sw_blocks blocks (modeling_arloop.py:131-148, :196-198);
    the key path (Kimi attn_res_block_size, kotodama boundaries, AliceAI, Cyrah) "-1/-1" (its unit, layers vs sublayers,
    unchecked: equal only among themselves, as before). "" without attn_res."""
    if not _r3("attn_res_keys_unread") or "attn_res" not in f:
        return ""
    mt = str(d.get("model_type") or "").lower()
    if mt == "gpts3_hybrid":
        return "1/0"
    if mt == "klm":
        return "2/0"
    if mt == "arloop" and _num(d.get("attnres_block_size")) > 0:
        return f"{_num(d.get('attnres_block_size'))}/{_num(d.get('sw_blocks'))}"
    return "-1/-1"


def _sig_covers(t, h):
    """A pair rule on a categorical signature (attn_res_sig / engram_sig / ve_kind): no constraint unless both points
    have one; then the target's value is one of the host's branches (a '|' set: the universal host), or the host's value
    is "*" (a branch whose geometry the loader sets from the target: the universal host's UEngram)."""
    return (not t) or (not h) or h == "*" or t in branches(h)


# CENO's attention / MoE / MLP blocks replace the stream by a norm of it (modeling_ceno.py:999-1062: "final output =
# pre_mlp_layernorm(x + attention(input_layernorm(x)))", the MoE / MLP residual is the normed input): post_ln
# (ceno_megatron_stream_renorm), whenever the pattern has a non-Mamba block
POST_LN_BY_CODE = _R3Set((), add={"ceno_megatron_stream_renorm": {"ceno"}})
# Dory's nGPT sphere residual (ngpt_sphere_residual): h <- L2(h) + r (L2(f) - L2(h)), every norm a weight-free L2
# (transformers-pr50099 modeling_dory.py:50-61, :451-483)
NGPT_TYPES = _R3Set((), add={"ngpt_sphere_residual": {"dory"}})
# CFRD's data-dependent scalar residual gate on both sublayers (cfrd_sigmoid_residual_gate; PLAN decision 4: res_gate)
RES_GATE_TYPES = _R3Set((), add={"cfrd_sigmoid_residual_gate": {"cfrd"}})
ZAYA_TYPES = frozenset({"zaya1_vl", "zaya", "zaya1"})


def _ceno_renorms(d):
    """A CENO pattern with an attention / MoE / MLP block (hybrid_override_pattern '*' / 'E' / '-', or layers_block_type
    attention / moe / mlp): the blocks whose output stream is a normed one."""
    pat, lbt = d.get("hybrid_override_pattern"), d.get("layers_block_type")
    return ((isinstance(pat, str) and any(c in pat for c in "*E-"))
            or (isinstance(lbt, list) and any(str(x) in ("attention", "moe", "mlp") for x in lbt)))


def _hc_feats():
    return {f"hc_{v}": ("wall", "wall",
                        f"hyper-connection variant {v} (round 3: hyper_conn_variant): the stream-mixing code of its "
                        f"family; no other variant's maps are a weight setting of it (Sinkhorn comb vs comb^T at finite "
                        f"iterations, identity carry, gated residual, lagged pre, learned vs mean collapse)")
            for v in HC_VARIANTS}


def _engram_feats():
    out = {}
    for k in ENGRAM_KINDS:
        host = "wall" if k == "engram_klm" else "free"
        why = ("its zeroed memory is still an AttnRes source (q . rms(0) = 0 keeps softmax mass): no switch-off"
               if host == "wall" else "the read projection 0 adds exactly 0 (Lean extra_input_off)")
        out[k] = ("wall", host,
                  f"an n-gram memory of the {k[7:]} code (round 3: its own hash / addressing / injection, compared with "
                  f"equal geometry, dims engram_sig): no token embedding or other hash reads the same previous-token "
                  f"rows; host side: {why}")
    return out


# round 3 (nonlm_kept; a design decision on not_a_causal_lm): the two score-head rerankers and sundial are NOT skipped.
# They stay placed with a feature each that is a wall BOTH ways against ordinary LMs, carried only by the universal
# host's join (minimal_emulator HOST_FEATS; its branches are built and tested on branch host-nonlm, not merged here):
#   score_head  the MiniCPM layer-wise reranker (head_type "simple": lm_head = ModuleList of LayerWiseHead(hidden, 1)
#               per exit layer, modeling_minicpm_reranker.py:1134-1185) and BAAI's Gemma-2 cost-wise reranker
#               (layer_wise: CostWiseHead(hidden, 1) per exit layer from start_layer, gemma_model.py:485-511, :589-600):
#               one relevance score per exit layer, not a vocabulary distribution. Target side: no LM's lm_head (vocab x
#               hidden, read after the last layer) emits the per-exit-layer scalar score. Host side: a score head emits
#               no vocabulary distribution, so a reranker plays no plain LM.
#   ts_patch    sundial (modeling_sundial.py:28-45 SundialPatchEmbedding: float value patches of 16 through a SiLU MLP,
#               no token table; :403 FlowLoss head, flow_loss.py:40-49: 50 Euler steps from Gaussian noise). Target side:
#               no token embedding forms a continuous patch input, no lm_head a flow-matching sample. Host side: sundial
#               reads no token ids and emits no next-token logits, so it plays no plain LM.
# (_not_lm's kinds score_head / continuous_io map to these two feature names.) These two features are the
# representation with nonlm_heads in R3_OFF only: with it on, the head / input SLOTS carry these models (HEAD_ORDER,
# INPUT_ORDER: the readout rule makes a score head exact on an untied LM host, the flow head and the patch input stay
# walls both ways) and _feats adds neither feature.
NONLM_FEAT = {"score_head": "score_head", "continuous_io": "ts_patch"}


def _early_exits(d, outer=None):
    """Round 3 (nonlm_heads): a score model whose saved config asks for exits other than the last layer (cutoff_layers
    not [num_hidden_layers]). The released default call reads ONE exit at L (cutoff_layers None -> [L]:
    modeling_minicpm_reranker.py:1256-1257, gemma_model.py:595-596); cutoff_layers is a call argument no snapshot config
    saves, so no snapshot model has the feature."""
    outer = outer if isinstance(outer, dict) else {}
    cl = d.get("cutoff_layers", outer.get("cutoff_layers"))
    if cl is None:
        return False
    L = _num(d.get("num_hidden_layers"))
    return any(_num(x) != L for x in (cl if isinstance(cl, list) else [cl]))


FEATURES = _R3Dict(FEATURES, add={
    # round 3 (nonlm_heads): EARLY EXITS, exit_heads (not per_token_early_exit's data-dependent early_exit) -- a
    # target-side feature. A target that outputs the stream after e < L layers (norm(x after e layers) -> its own
    # head) needs a host with exits: a host without them overwrites the normed
    # intermediate state, and a copy carried in spare channels is renormalised jointly by the final norm (the
    # post_norm / hyper_conn argument, prenorm_readout_obstruction). Host side free: a host with exits plays a plain
    # target by reading only the exit at L (its other exit heads are outputs nobody reads; catalog r3_exit_heads_off).
    # Binds no snapshot model at the released default call (_early_exits)
    "nonlm_heads": {
        "exit_heads": ("wall", "free",
                       "early exits (a readout of norm(x after e < L layers) through its own head: the rerankers' "
                       "cutoff_layers): a host without exits cannot output an intermediate normed state (a carried copy "
                       "meets the final norm); a host with exits reads only the one at L")},
    "nonlm_kept": {
        "score_head": ("wall", "wall",
                       "a non-vocab output head (the rerankers' Linear(hidden, 1) per exit layer: "
                       "modeling_minicpm_reranker.py:1134-1185, gemma_model.py:485-511): one relevance score per exit "
                       "layer, no next-token distribution; no LM's lm_head forms it, and a reranker emits no vocab "
                       "logits for a plain LM target (the universal host's branch: host-nonlm)"),
        "ts_patch": ("wall", "wall",
                     "continuous patch input and a flow-matching head (sundial: SundialPatchEmbedding of float value "
                     "patches, modeling_sundial.py:28-45; FlowLoss sampling, :403): no token embedding / lm_head of an "
                     "LM forms them, and sundial reads no token ids nor emits logits for a plain LM target (the "
                     "universal host's branch: host-nonlm)")},
    "hyper_conn_variant": _hc_feats(),
    "qwen4_ple_reads_ngrams": {"engram_qwen4": _engram_feats()["engram_qwen4"]},
    "ngram_memory_unread": {k: v for k, v in _engram_feats().items() if k != "engram_qwen4"},
    "residual_gain_vector": {
        # target: needs theorem res_gain_not_fold (RMSNorm(D x) / RMSNorm(x) is not constant for a non-uniform diagonal
        # D; the uniform case is stream_scalar_needs_eps). Host: g = 1 (Doge input_residual / post_attention_residual;
        # ZAYA s = 1) is the plain residual, the identity setting (catalog feature:res_gain:host)
        "res_gain": ("wall", "free",
                     "a per-channel learned gain on the residual before each merge, h = g (.) r + f(N r) (Doge "
                     "modular_doge.py:389-427; ZAYA1-VL ResidualScaling (r + b) (.) s, modeling_zaya1_vl.py:1512-1531): "
                     "the next pre-norm sees D x, which no gain-free stream forms; a host sets g = 1"),
        # ZAYA's residual bias b: a host sets b = 0 (Lean bias_off on the residual merge; catalog feature:res_bias:host)
        "res_bias": ("wall", "free",
                     "a per-channel learned bias on the residual before each merge (ZAYA1-VL (r + b_r) s_r): a "
                     "constant the gain-only / plain stream cannot add under its pre-norms; a host sets b = 0"),
        # Muse-Spark's coupled gate: alpha^2 = 1 - beta^2 above the clamp, so alpha = 1 forces beta -> 0 (the sublayer
        # off): a wall both ways (PLAN decision 4: res_gate_ms, not CFRD's res_gate)
        "res_gate_ms": ("wall", "wall",
                        "Muse-Spark's residual gate alpha (.) r + beta (.) h, beta = sigmoid(w / T), alpha = sqrt(sigmoid"
                        "(-w / T)(1 + beta)) (modular_muse_spark.py:451-462): the two coefficients are coupled per "
                        "channel; no setting gives alpha = beta = 1")},
    "backout_residual": {
        # target: OPEN (as "engram / ple / value_emb target <- plain host"). Host: backout_lambda = 0, x_L - 0 x_m = x_L
        # (Lean extra_input_off: an added read of an earlier stream through a scalar set to 0; catalog
        # feature:backout:host)
        "backout": ("wall", "free",
                    "backout: a scaled mid-layer residual subtracted before the final norm, x_L - lambda x_m (Gaggle "
                    "modeling_gaggle.py:293-311, m = n_layer // 2; nanochat_gpt backout_layer, modeling_nanochat_gpt.py:"
                    "343-344, :492-496): no single stream carries both weights of the early writers; a host sets "
                    "lambda = 0")},
    "ngpt_sphere_residual": {
        # needs theorem ngpt_not_additive (both sides; cf. selfcond_norm_not_exact: an always-on weight-free norm)
        "ngpt": ("wall", "wall",
                 "nGPT sphere residual (Dory): h <- L2(h) + r (.) (L2(f) - L2(h)) with weight-free L2 norms on the "
                 "residual and every sublayer output (modeling_dory.py:451-483): no additive pre-norm / sandwich / "
                 "post-LN stack keeps its stream on the sphere, and a Dory host cannot leave it")},
    "cfrd_sigmoid_residual_gate": {
        # host: the gate's weight and bias 0 (sigma = 1/2 on every token) and the sublayer's output projections x 2 / c:
        # Lean out_gate_off (W_g = 0, 2 W_o: the gate a broadcast row of the elementwise one) with the gate bias 0
        # (bias_off) and the residual_scale c folded (scalar_fold); catalog feature:res_gate:host
        "res_gate": ("wall", "free",
                     "a data-dependent scalar gate on each residual update, x + c sigmoid(w . u + b) f(u) (CFRD "
                     "FoldedCell / BindingBlock, haru_1.1/model.py:508-558): no ungated layer forms the product (an "
                     "out_gate host's gate covers attention only and has no bias); a gated host sets w = 0, b = 0 "
                     "(gate 1/2) and doubles the output projections")},
    "bolmo_byte_patch_hierarchy": {
        # needs theorem (both sides OPEN): no fixed-depth token-level stack forms the content-dependent pool / depool
        "byte_patch": ("wall", "wall",
                       "byte-latent hierarchy (Bolmo): a byte encoder / decoder with mLSTM local layers around a global "
                       "transformer on content-pooled patches (modeling_bolmo.py:599-870, :1050-1158); no token-level "
                       "stack forms the pool / depool, and the boundary predictor and local layers have no off switch")},
})

# --- fr-resnorm: residual streams and norms paid (dist/fr_resnorm.md; Lean ResNorm.lean) ------------------------------
# HYPER-CONNECTION HOSTS SWITCHED OFF (fr_hc_host_off). Every stream carries the target's residual: the embedding enters
# as n copies (every stock HC model expands it so), the dynamic projections are 0 (fn / hc_fn; Qwen4-exp's
# input_mix_weight_up and block_inject_weight), and the static bases give sum_i pre_i = 1, post_i = 1 and comb rows
# summing to 1 -- identical streams then get the plain update x + f(x) for ANY sublayer (Lean hc_identical_streams,
# hc_stack: exact). Per stock code (transformers 5.18; hf/hc_host_off.py checks each module in fp64):
#   dsv4 / glm5_next  the Sinkhorn of constant comb logits is exactly 1/n (sinkhorn_const) only with hc_eps = 0 -- a
#                     shape-preserving config value (free); with hc_eps > 0 the rows sum to < 1 (sinkhorn_eps_rowsum_lt;
#                     catalog fr_hc_dsv4_eps, the finding); V4's HyperHead base logit(1/n), GLM-5-Next's mean;
#   hy4               no comb: sigmoid(pre) = 1/n - hc_eps, sigmoid(post) = (1 - hc_eps) / hc_magnitude, any hc_eps < 1/n;
#   qwen4_exp         the mix gate sigmoid(0) = 1/2 against the per-stream (1 + w) norm gain 2 gamma (that norm IS the
#                     pre-norm), the injection 2 sigmoid(0) = 1, the final mixer the final norm.
# Measured on real checkpoints (hf/hc_host_agreement.py, hf/data/hc_host_agreement.jsonl): gemma-2b and codegemma-2b on
# a Hy4 host, Qwen3-0.6B and Qwen3-1.7B on a Qwen4-exp host. The other variants' codes (dsv4_nano, dsv41's lagged pre,
# xing, motif, needle, san) are remote and unchecked here: their hc_<variant> host sides stay walls, so every
# hyper-connection host still needs its variant's switch-off (each HC point carries its hc_<variant> key). The target
# sides stay walls: a plain host carrying n streams in n x width reads them through ONE pre-norm over all n d channels
# and with fixed weights, while the target's sublayer reads a data-dependent mix normed on its own (refuted by
# hf/hc_host_off.py's width check, dist/fr_resnorm.md).
HC_HOST_OFF_VARIANTS = ("dsv4", "glm5_next", "hy4", "qwen4_exp")
# DIFFUSIONGEMMA'S ALWAYS-ON EMBEDDING NORM SWITCHED OFF (fr_selfcond_norm_host). The whole residual stream scaled by
# alpha with host eps = alpha^2 eps (a config value): every norm whose input is the stream or a linear image of it (the
# pre-norms, the final norm, the router's norm; the qk- / v- / post- / branch norms with their writers x alpha) gives the
# target's output exactly (Lean rmsNorm_smul_eps_sq); the embedding norm returns alpha e from the row lambda_v e,
# lambda_v = alpha^2 sqrt(eps) / sqrt(1 - alpha^2 ms(e_v)), for every row once alpha^2 ms(e_v) < 1 (selfcond_norm_row --
# the old escape selfcond_norm_escape needed ms(e_v) < 1 of the target itself). Exact on an untied head; DiffusionGemma
# ties it, so logit v is x 1 / sqrt(1 - alpha^2 ms(e_v)) -> 1: a LIMIT (alpha -> 0). Credited only where the code is
# read: the stock DiffusionGemma (modeling_diffusion_gemma.py:790-823, its post_norm eps = rms_norm_eps, the one config
# eps every norm shares) -- dims sc_norm_off, a host_off of the feature check in the payable modes. QwenDLM-ORL's
# modeling_sdar.py is not public: its norm stays a wall on both sides.
SC_NORM_OFF_TYPES = frozenset({"diffusion_gemma", "diffusion_gemma_text"})


def _fr_host_off(f, sc_norm_off, payable):
    """fr-resnorm host switch-offs of a host feature string f: a hyper-connection host of a checked variant switches
    its streams off (exact: every mode that reads features; port hc_off), a DiffusionGemma host its embedding norm (a
    limit: the payable modes only; port sc_norm_off)."""
    fs = set(filter(None, (f or "").split("+")))
    out = set()
    if _r3("fr_hc_host_off") and _p5("hc_off"):
        hv = [v for v in HC_HOST_OFF_VARIANTS if f"hc_{v}" in fs]
        if hv:
            out |= {"hyper_conn"} | {f"hc_{v}" for v in hv}
    if _r3("fr_selfcond_norm_host") and _p5("sc_norm_off") and sc_norm_off and payable:
        out.add("selfcond_norm")
    return frozenset(out & fs)


def _loop_count(d, mt):
    """The loop count num_loops-style keys give (qwen3loop: len(loop_loss_weights) + 1 when listed, 1 under the double
    split: get_num_loops, modeling_qwen3loop.py:89-95)."""
    if d.get("enable_double_loop_split") is True:
        return 1
    w = d.get("loop_loss_weights")
    if mt == "qwen3loop" and isinstance(w, list) and w:
        return len(w) + 1
    return _num(d.get("num_loops"))


def _loop_feats(d, mt):
    """Round 3, batch 4a: the loop-boundary FEATURES of an LM config d (loop_boundary_norm, loop_input_injection,
    loop_kv_share_gated_mix, hrm_hierarchical_loop, per_token_early_exit, lfm3rd_recurrent_controller)."""
    f = set()
    if _r3("loop_boundary_norm"):
        if mt in ("nanbeige", "qwen3loop"):
            if d.get("skip_loop_final_norm") is not True and _loop_count(d, mt) > 1:
                f.add("loop_bnorm_cfg")
        elif ((mt == "ouro" and (_num(d.get("total_ut_steps")) or 4) > 1) or mt == "hrm_text"
              or (mt == "dory" and _num(d.get("n_recurrent_loops")) >= 1 and _num(d.get("n_recurrent_layers")))):
            f.add("loop_bnorm")
    if _r3("loop_input_injection") and (
            (mt == "dory" and _num(d.get("n_recurrent_loops")) >= 1 and _num(d.get("n_recurrent_layers")))
            or (mt in ("huginn_raven", "huginn", "raven") and _num(d.get("mean_recurrence")) >= 1)
            or (mt == "lfm3_rd" and _num(d.get("recurrent_loops")) > _num(d.get("num_motifs")) > 0)):
        f.add("loop_inject")
    if _r3("loop_kv_share_gated_mix") and mt == "iquestloopcoder" and (_num(d.get("loop_num")) or 2) > 1:
        f.add("loop_kv_mix")
    if _r3("hrm_hierarchical_loop") and mt == "hrm_text":
        f.add("hrm_two_state")
    if _r3("per_token_early_exit") and (
            (mt == "ouro" and (_num(d.get("total_ut_steps")) or 4) > 1 and d.get("early_exit_step") is None
             and d.get("early_exit_threshold") is not None)
            or (mt == "lfm3_rd" and "auto" in (str(d.get("reasoning_mode")), str(d.get("reasoning_effort"))))):
        f.add("early_exit")
    if _r3("lfm3rd_recurrent_controller") and mt == "lfm3_rd" and _num(d.get("num_motifs")) \
            and _num(d.get("recurrent_loops")):
        f.add("rd_controller")
    return f


def _q_temp(d, mt):
    """The ungated query temperatures (position_query_temperature): "q_temp" (Mistral-Large-4's learned per-head
    log-position scale), "q_temp_nope" (Beam's code-fixed one on the NoPE full layers), else None."""
    if not _r3("position_query_temperature"):
        return None
    if mt in ("mistral_large4", "mistral_large4_text"):
        return "q_temp"
    if mt == "beam" and any(v is None for v in _rope_layer_types(d.get("rope_parameters")).values()):
        return "q_temp_nope"
    return None


def _pos_gate(d, outer=None):
    """Round 3 (PLAN decision 3, context gates): the first position where a position-gated change of the attention
    starts (0 = none), the min over the model's gates. Below it the attention IS the plain one (the factor 1, the
    configured table), so the headline (USE_CONTEXT off) credits the target unchanged -- the DSA-as-dense precedent:
    a design decision caps the credited context at the gate (emul_matrix's USE_CONTEXT block), and the caveat is that
    no edge into such a target holds beyond it (dims pos_gate, keyed only under USE_CONTEXT):
      position_query_temperature  Qwen-1 use_logn_attn: q * log_{seq_length}(i) past seq_length (modeling_qwen.py
          :289-294, :482-490); Ministral-3 / Mistral-4 / Nemotron-Labs-Diffusion q * (1 + beta log(1 + floor(pos /
          original_max_position_embeddings))) (tf5 modeling_ministral3.py:105-107, :146-150; modeling_mistral4.py:370-372,
          :489-493; Mistral4Config default beta 0.1); Llama-4 attn_temperature_tuning on its NoPE layers past floor_scale
          (tf5 modeling_llama4.py:321-323, :367-374; class defaults True / 0.1 / 8192)
      dynamic_ntk_in_context  Qwen-1 use_dynamic_ntk: the base grows with the prompt length past seq_length (modeling_
          qwen.py:735-739, :833-850, :1287-1298); Meditor-1 / -2: a code-fixed threshold 32768 / 8192 on the global
          layers (modeling_meditor.py:60-85, modeling_meditor2.py:71-93)"""
    mt = str(d.get("model_type") or "").lower()
    gates = []
    if _r3("position_query_temperature"):
        if mt == "qwen" and d.get("use_logn_attn") is True:
            gates.append(_num(d.get("seq_length")) or 8192)
        if mt in ("ministral3", "mistral4", "nemotron_labs_diffusion"):
            rp = d.get("rope_parameters") or d.get("rope_scaling") or {}
            rp = rp if isinstance(rp, dict) else {}
            beta = rp.get("llama_4_scaling_beta")
            if beta is None and mt == "mistral4" and not (d.get("rope_parameters") or d.get("rope_scaling")):
                beta = 0.1                                 # Mistral4Config's default rope_parameters
            try:
                beta = float(beta) if beta is not None else 0.0
            except (TypeError, ValueError):
                beta = 0.0
            if beta:
                gates.append(_num(rp.get("original_max_position_embeddings")) or 8192)
        if mt in ("llama4", "llama4_text"):
            try:
                sc = float(d.get("attn_scale", 0.1) if d.get("attn_scale") is not None else 0.1)
            except (TypeError, ValueError):
                sc = 0.1
            nrl = d.get("no_rope_layers")
            has_nope = isinstance(nrl, list) and any(not x for x in nrl)
            if d.get("attn_temperature_tuning", True) is not False and sc != 0 and has_nope:
                gates.append(_num(d.get("floor_scale")) or 8192)
    if _r3("dynamic_ntk_in_context"):
        am = _auto_map_str(d, outer)
        if mt == "qwen" and d.get("use_dynamic_ntk") is True:
            gates.append(_num(d.get("seq_length")) or 8192)
        if mt == "meditor" and "modeling_meditor" in am:
            gates.append(32768)
        if mt == "meditor2" and "modeling_meditor2" in am:
            gates.append(8192)
    gates = [g for g in gates if g > 0]
    return min(gates) if gates else 0


def _fope(d):
    """Intern-S1-Pro's FoPE switch: any fope key in rope_scaling / rope_parameters (fope_positions)."""
    return any(isinstance(x, dict) and any(x.get(k) is not None for k in ("fope_init_factor", "fope_sep_heads",
                                                                          "num_inv_freq"))
               for x in (d.get("rope_scaling"), d.get("rope_parameters")))
# FEAT_NEST (PLAN decision 1): a target feature x is carried by a host that has x or a listed parent p, with kind "exact"
# (every mode) or "limit" (outside the exact slice); the parent p the host spends on x is not switched off (feats_ok).
# Entries are keyed by spec; an entry needs its theorem (PROOFS.md) or a measurement. The candidates without one stay
# walls in FEAT_NEST_PENDING (name -> the theorem it waits for), off the relation.
FEAT_NEST = _R3Dict({})
FEAT_NEST_PENDING = {
    ("out_gate_head", "out_gate"): "out_gate_head_sub_elem",
    ("out_gate_head_softplus", "out_gate_softplus"): "out_gate_head_sub_elem",
    ("out_gate_head_silu", "out_gate_silu"): "out_gate_head_sub_elem",
    ("out_gate_head_bias", "out_gate_bias"): "out_gate_head_sub_elem",
    ("out_gate", "out_gate_bias"): "out_gate_zero_bias",
    ("out_gate_head", "out_gate_head_bias"): "out_gate_zero_bias",
    # round 3, batch 2d (xsa_self_value_projection): a unit-XSA target on Limite's scaled host is the tanh(alpha) -> 1
    # limit (bounded |y|), neither proved nor measured
    ("xsa", "xsa_scaled"): "xsa_alpha_limit",
    # round 3, batch 4b (hyper_conn_variant): a glm5_next target (mean collapse) on a dsv4 host -- the learned HyperHead
    # with hc_fn = 0 and base = logit(1 / hc_mult - hc_eps) is the mean (elementary, but no theorem / catalog instance)
    ("hc_glm5_next", "hc_dsv4"): "hc_head_mean",
}


# Round 3 (mamba1_lever; branch fr-mixer's rule, dist/fr_mixer.md, applied here with mamba_ffn_none): a plain Mamba-1
# target on a host whose dt / B / C pass RMS norms (Falcon-Mamba: weight-free, eps mixer_rms_eps; Jamba: weighted, eps
# rms_norm_eps). Those inputs ARE linear maps' outputs (x_proj rows), so the weights-only eps lever makes the three
# norms linear: x_proj rows x 2^-a; Jamba's gains 2^a sqrt(e) (Lean epsLever_eq); Falcon-Mamba's weight-free constant s
# = 2^-a / sqrt(e) folds into dt_proj (x 1 / s), D (x s^2) and out_proj (x 1 / s^2) (weightFree_lever,
# weightFree_lever_dot, lever_pair_factor; proofs/PROOFS.md "mamba1_lever"). A HOST WEIGHT CONSTRUCTION (no
# config value moves), a LIMIT: payable, never exact. MEASURED (hf/mixer_rules_agreement.py,
# hf/data/mixer_rules_agreement.jsonl, fp32, 24 prompts): mamba-130m on Falcon-Mamba a = 28 top-1 100% / 95.8% of
# 32-token continuations identical, mamba-370m 100% / 100%, mamba-130m on Jamba (8 zeroed attention layers) a = 27
# 100% / 95.8%. Dims RC_GE (padding is inert once the norms are linear); the host needs eps > 0 on those norms
# (eps_lever). The Jamba nest also needs a pure-on-hybrid mixer edge (ssm <- hybrid), which this relation does not
# grant: it credits no edge until that rule lands.
# branch fr-mixer, rule m2_nonorm_lever (dist/fr_mixer.md; Lean FrMixer.gated_norm_residual_lever): a Mamba-2 target
# WITHOUT the gated output norm (Falcon-H1 mamba_rms_norm false: y * silu(z)) on a host WITH one (Mamba-2 / Nemotron-H /
# Bamba / Granite-4.0-H / Falcon-H1: RMSNorm(y * silu(z)), or Falcon-H1's norm-before-gate RMSNorm(y) * silu(z)). The
# norm's input is a silu / scan output, so no weight scales it (the weights-only eps lever cannot reach it); the config
# eps can: the residual stream x S (embedding, every sublayer's output projection) with host eps = e_t S^2 keeps every
# stream norm exactly the target's (rmsNorm_smul_eps), and the gated norm, fed an O(1) input against eps e_t S^2, is
# the linear map w / (S sqrt(e_t)) (epsLever_eq: a LIMIT, at the fp32 rounding floor once ms / (e_t S^2) < 2^-25) --
# gain S sqrt(e_t) makes it the identity, the target's out_proj x S writes the scaled stream. MEASURED 100% / 100%
# (hf/mixer_rules_agreement.py). Payable, not exact; only where the gated norm's eps IS the config eps
# (REC_EPS_CFG_TYPES; not Zamba2's 1e-5 in code) and the host can run the lever (eps_lever): _rec_ok.
# (Falcon-H1's norm-before-gate host, RMSNorm(y) silu(z), runs the same lever, but its family is not rec_eps_cfg in the
# relation and the zoo has none: not nested.)
R3_FEAT_NEST = {"rf_mamba1": (("rf_mamba1_bcdt", "limit"), ("rf_mamba1_bcdtw", "limit")),
                "rf_mamba2_nonorm": (("rf_mamba2", "limit"),)}
_R3_NEST_SPEC = {"rf_mamba1": "mamba1_lever", "rf_mamba2_nonorm": "m2_nonorm_lever"}
# the fr-mixer rules' switches (pure_on_hybrid, m2_nonorm_lever, mamba1_lever) are round-3 spec keys (R3_OFF)
FRM_OFF = R3_OFF


def _frm(rule):
    return _r3(rule)


def _r3_nest(x):
    return R3_FEAT_NEST.get(x, ()) if x in _R3_NEST_SPEC and _r3(_R3_NEST_SPEC[x]) else ()


def _feat_parents(x, strict):
    """The host features that carry target feature x by a FEAT_NEST entry (exact ones only under strict); round 3:
    + the recurrent lever nests (R3_FEAT_NEST, limits)."""
    nest = tuple(FEAT_NEST.get(x, ())) + tuple(_r3_nest(x))
    return tuple(p for p, kind in nest if kind == "exact" or not strict)


def _feats(d, outer=None):
    """The FEATURES a config's forward pass uses (d = the LM config, outer = the wrapper it came from)."""
    outer = outer or {}
    mt = str(d.get("model_type") or "").lower()
    get = lambda k: d.get(k, outer.get(k))
    f = set()
    head = _readout(d, outer)[0]
    if _tied(d, outer) and head == "lm":
        # (round 3, nonlm_heads: a model whose output is not the lm_head -- a score / value / flow head, a pooled
        # embedding -- has no output table to tie: tie_word_embeddings is unread for its readout)
        f.add("tied_emb")
    og = _out_gate_key(d, mt)
    if og:
        f.add(og)
    if mt == "granite_switch" or d.get("num_adapters") or d.get("adapter_token_ids"):
        f.add("switch_lora")
    if (_num(d.get("hc_mult")) > 1 or _num(d.get("hc_count")) > 1 or d.get("mhc") or d.get("mhc_enabled")
            or _num(d.get("mhc_expansion_rate")) > 1 or _num(d.get("hc_rate")) > 1 or d.get("hc_magnitude")
            # round 3 (hyper_conn_key_unread): Cactus needle's extras.mhc_lanes / SAN's mhc_lanes (code default 4)
            or (_r3("hyper_conn_key_unread") and (_mhc_lanes(d) or 0) > 1)):
        f.add("hyper_conn")
        hv = _hc_variant(d, outer) if _r3("hyper_conn_variant") else None
        if hv:
            f.add("hc_" + hv)                              # round 3 (hyper_conn_variant)
    if d.get("engram_layer_ids") or d.get("engram_vocab_size"):
        f.add("engram")
    ek, _ = _engram_kind(d)                                # round 3 (qwen4_ple_reads_ngrams / ngram_memory_unread)
    if ek:
        f.add(ek)
    # (round 3, qwen4_ple_reads_ngrams: Qwen4-exp's ple_layer_ids / ple_embed_dim are its n-gram memory above, not PLE)
    q4 = _r3("qwen4_ple_reads_ngrams") and mt in QWEN4_TYPES
    if _num(d.get("hidden_size_per_layer_input")) > 0 or (not q4 and (d.get("ple_layer_ids")
                                                                      or _num(d.get("ple_embed_dim")) > 0)):
        f.add("ple")
    if (d.get("value_embedding_layers") or _num(d.get("ve_gate_channels")) or d.get("use_x0_lambdas")
            or d.get("use_smear") or d.get("value_residual") or _ve_by_code(d)):
        f.add("value_emb")
    if (_num(d.get("attn_res_block_size")) > 0 or _num(d.get("block_attn_res_block_size")) > 0
            or d.get("attn_res_boundaries") or d.get("attention_residuals")
            # round 3 (attn_res_keys_unread): ARLoop's attnres_block_size spelling, GPT-S3 / KLM by code
            or (_r3("attn_res_keys_unread") and (_num(d.get("attnres_block_size")) > 0
                                                 or mt in ("gpts3_hybrid", "klm")))):
        f.add("attn_res")
    # round 3, batch 4b (residual streams): Doge / ZAYA residual gains, ZAYA's residual bias, Muse-Spark's coupled gate,
    # the backout path, Dory's nGPT sphere, CFRD's scalar residual gate, Bolmo's byte-patch hierarchy
    if _r3("residual_gain_vector"):
        if mt == "doge" or (mt in ZAYA_TYPES and _truthy(d.get("scale_residual_merge"))):
            f.add("res_gain")
            if mt in ZAYA_TYPES:
                f.add("res_bias")
        if mt in ("muse_spark", "muse_spark_text"):
            f.add("res_gate_ms")
    if _r3("backout_residual") and (mt == "gaggle" or (mt == "nanochat_gpt" and d.get("backout_layer") is not None)):
        f.add("backout")
    if mt in NGPT_TYPES:
        f.add("ngpt")
    if mt in RES_GATE_TYPES:
        f.add("res_gate")
    if _r3("bolmo_byte_patch_hierarchy") and mt == "bolmo":
        f.add("byte_patch")
    if mt in ONORM_TYPES:
        f.add("onorm")                                     # round 3 (attn_output_head_norm)
    if mt.startswith(("gemma4", "diffusion_gemma")) or d.get("value_norm") is True or mt in VNORM_TYPES:
        f.add("vnorm")
    # several RoPE tables (Gemma-3 rope_local_base_freq, MiMo / dots3 swa_rope_theta, rope_parameters per
    # layer type) are no feature: the rot gate (rope_fits on dims rope_sched / rope_cap) prices them
    if _nope_layer_mix(d):
        f.add("nope_layers")
    # round 3, batch 3a (router_feats_on_dense_model): router features only where a routed layer is built (Cyrah's one
    # expert / first_k_dense_replace = L, modeling_cyrah.py:532-533; NemotronH lists with no 'moe' entry: _moe_disabled)
    moe_built = (not _r3("router_feats_on_dense_model")
                 or (not _moe_disabled(d) and max(_num(d.get(k)) for k in ("num_experts", "moe_num_experts",
                                                                         "n_routed_experts", "num_local_experts")) > 1)
                 or _hier_moe(d))
    if (_moe_like(d) or d.get("num_local_experts") or d.get("n_routed_experts") or _hier_moe(d)) and moe_built:
        sig, ssp, bias = _router_flags(d)                                           # the router of a MoE
        if sig:
            f.add("router_sigmoid")
        if ssp:
            f.add("router_sqrtsoftplus")
        if bias:
            # round 3, batch 3a (router_bias_unread): a logit-domain selection bias is its own feature
            f.add("router_bias_logit" if _router_bias_logit(d) else "router_bias")
        if mt in ("llama4", "llama4_text"):
            f.add("router_in_gate")
        if (_dsv4_schema(d, mt)[1] if _r3("dsv4_layer_types_schema") and mt in DSV4_SCHEMA_TYPES
                else _num(d.get("num_hash_layers")) > 0):
            f.add("hash_route")
        if (_r3("hash_route_by_code") and mt == "deep"
                and str(d.get("mlp_type", "token_routed")).lower() == "token_routed"):
            # round 3, batch 3a: the deep family's token-routed MLP (modeling_deep.py:213-229: routes =
            # topk_token_to_expert[:, token_ids], constant route weights; DeepConfig refuses other strategies)
            f.add("hash_route")
        blk = _block_routing(d)
        if blk:
            f.add("block_routing")                     # (round 3: the gate never reads n_group / topk_group)
        elif 1 < _num(d.get("n_group")) and 0 < _num(d.get("topk_group")) < _num(d.get("n_group")):
            f.add("router_group")
        if (_r3("router_group_balanced") and mt in ("interns1_pro", "interns1_pro_text")
                and _num(d.get("router_n_groups")) > 1):
            f.add("router_group")                      # round 3: Intern-S1-Pro's per-group router
        if mt in SHARED_GATE_TYPES and (d.get("shared_expert_intermediate_size") or mt == "qwen3_next"):
            f.add("shared_gate")
        if _r3("mlp_router") and mt == "llama_moe" and str(d.get("gate_network", "mlp")).lower() == "mlp" and (
                str(d.get("gate_type", "TopKBalancedNoisyGate")).lower() == "topkbalancednoisygate"):
            f.add("router_mlp")
        if _r3("moe_capacity_drop") and mt == "hunyuan_v1_moe" and "hunyuan.HunYuan" in _auto_map_str(d, outer):
            f.add("moe_capacity")
        if _r3("expert_output_norm") and mt in ("muse_spark", "muse_spark_text"):
            f.add("expert_norm")
        if _r3("gemma4_branch_post_norms") and _dense_mlp_beside_moe(d):
            f.add("moe_branch_norm")
        if (_r3("arctic_dense_residual_parallel_moe") and mt == "arctic" and d.get("parallel_attn_mlp_res") is True
                and _num(d.get("num_local_experts")) > 1):
            f.add("par_moe_res")
    if ((mt in BIAS_QKV_TYPES and d.get("use_qkv_bias", d.get("qkv_bias", True)) is not False)
            or any(d.get(k) is True for k in ("attention_bias", "use_qkv_bias", "qkv_bias", "add_qkv_bias", "bias"))
            or (mt in ("falcon",) and d.get("bias") is True) or (mt == "mpt" and d.get("no_bias") is False)):
        f.add("bias_qkv")
    if (mt in BIAS_OUT_TYPES                                 # (Falcon: the LayerNorm beta is there even with bias false)
            or any(d.get(k) is True for k in ("mlp_bias", "use_bias", "add_bias_linear", "enable_bias"))
            or (mt == "mpt" and d.get("no_bias") is False)
            # one flag that also biases o_proj (LlamaAttention and its descendants: o_proj bias=config.attention_bias;
            # InternLM-1 / 2 / 3: bias=config.bias). Qwen2 / GLM / Seed-OSS / MiMo bias only q / k / v.
            or (mt in O_BIAS_FROM_ATTN_BIAS and d.get("attention_bias") is True)
            or (mt in ("internlm", "internlm2", "internlm3") and d.get("bias") is True)
            or (mt == "rwkv7" and d.get("norm_bias", True) is not False)):     # fla RWKV-7 LayerNorm(bias=norm_bias)
        f.add("bias_out")
    # round 3 (norm_placement_keys_unread): placement keys p5 did not read. Molmo norm_after true and Bolmo's
    # reordered_norm block (its default) are post-ONLY (modeling_molmo.py:478-527 `if not norm_after` pre-norm / `if
    # norm_after` output norm; modeling_bolmo.py:310-333 pre_norm = block_type != "reordered_norm"). MiniMax's residual IS
    # the normed stream (tf5 modular_minimax.py:398-413 unconditionally; remote MiniMax-Text-01 / M1 iff postnorm,
    # modeling_minimax_text_01.py:963-1044): post_ln, not a sandwich (the M2 arch is plain pre-norm, mlx-lm minimax.py:220).
    # IQuest-Q1 discards the raw stream (x = attention_norm(h_in); h = x + attn_out_norm(attn(x)) * s_a, model.py:498-505):
    # post_ln with an attention post-norm; Needle post-norms the attention output only (architecture.py:363-370)
    r3np = _r3("norm_placement_keys_unread")
    archs = [str(a) for a in (d.get("architectures") or outer.get("architectures") or [])]
    post_ln_mm = r3np and ((mt == "minimax" and not any("MiniMaxM2" in a for a in archs))
                           or (mt in ("minimax_text_01", "minimax_m1") and _truthy(d.get("postnorm"))))
    if mt in POST_ONLY_TYPES or (r3np and ((mt == "molmo" and d.get("norm_after") is True)
                                           or (mt == "bolmo" and (d.get("block_type") or "reordered_norm")
                                               == "reordered_norm"))):
        f.add("post_only")
    elif not post_ln_mm and (mt in POST_NORM_TYPES or d.get("use_post_norm") is True
                             or d.get("use_sandwich_norm") is True or d.get("post_norm") is True
                             or d.get("postnorm") is True):
        f.add("post_norm")
    if ((mt == "opt" and d.get("do_layer_norm_before") is False) or mt == "openai-gpt" or post_ln_mm
            or (r3np and mt == "iquest_q1") or (mt in POST_LN_BY_CODE and _ceno_renorms(d))):
        f.add("post_ln")
    if r3np and mt in ("needle", "iquest_q1"):
        f.add("post_norm_attn")
    cr = d.get("compress_ratios")
    no_compressor = isinstance(cr, list) and cr and not any(_num(x) for x in cr)   # DeepSeek-V4 with no compressed layers
    if _r3("dsv4_layer_types_schema") and mt in DSV4_SCHEMA_TYPES:
        no_compressor = not _dsv4_schema(d, mt)[0]         # (round 3: all sliding = no compressor, in either schema)
    sp = d.get("sparse_config") if isinstance(d.get("sparse_config"), dict) else {}
    if (((d.get("index_topk") or d.get("index_n_heads") or d.get("indexer_budget") or d.get("index_top_k"))
         and d.get("enable_dsa", True) is not False and not no_compressor)
            or (d.get("sparse_attention_config") or {}).get("use_sparse_attention")
            or d.get("moba_topk")                                  # MoBA block-sparse attention (Ring-mini-sparse)
            or (mt == "minicpm_sala" and sp.get("topk"))):         # InfLLM-v2 top-k blocks on SALA's softmax layers
        f.add("dsa")
    if (_dsv4_schema(d, mt)[0] if _r3("dsv4_layer_types_schema") and mt in DSV4_SCHEMA_TYPES
            else isinstance(cr, list) and any(_num(x) for x in cr)):
        # DeepSeek-V4 CSA / HCA compressed KV pools; round 3 (kv_compress_variant): V4.1's own compressor
        f.add("kv_compress_v41" if _r3("kv_compress_variant") and mt in ("deepseek_v41", "deepseek_v41_text")
              else "kv_compress")
    if d.get("clip_qkv"):
        f.add("clip_qkv")
    lim = d.get("swiglu_limit") or d.get("swiglu_limits") or d.get("swiglu_limits_shared")
    if not _num(lim) and _r3("act_clip_per_layer_list"):
        # round 3, batch 3b (act_clip_per_layer_list): Ling-3's BailingMoeV3SparseMoeBlock reads per-layer lists,
        # `limit_list[layer_idx] or None` (0 = no clamp), for the routed and the shared experts
        # (modeling_bailing_moe_v3.py:420-449; the post-activation clamp :342-348, act_clip_order: it converts exactly)
        lim = d.get("expert_swiglu_limit_list") or d.get("share_expert_swiglu_limit_list")
        lim = max(_num(d.get("expert_swiglu_limit_list")), _num(d.get("share_expert_swiglu_limit_list"))) if lim else lim
    # round 3, batch 3b (b_act_clip_host): act_clip only for the families whose code reads the limit (ACT_CLIP_CFG_TYPES:
    # K-EXAONE's swiglu_limits is never read by tf5 exaone_moe -- an unclamped SwiGLU under the HF code, PLAN decision 5)
    if (_num(lim) > 0 and mt not in ("gpt_oss", "gpt_oss_puzzle") and "oai" not in raw_act(d)
            and (mt in ACT_CLIP_CFG_TYPES or not _r3("b_act_clip_host"))):
        f.add("act_clip")
    if mt in FFN_NORM_TYPES:                               # round 3, batch 3b (ffn_inner_norm): Dory / BitNet
        f.add("ffn_norm")
    if d.get("use_self_conditioning") or mt.startswith("diffusion_gemma") or any(
            str(k).startswith("self_conditioning") for k in d):
        f.add("selfcond")
        if mt.startswith("diffusion_gemma") or d.get("self_conditioning_post_norm"):
            f.add("selfcond_norm")
    if mt in ("zamba", "zamba2"):
        f.add("zamba_shared")
    # --- round 3, batch 1 (FEATURES entries above, each keyed by its spec)
    if _r3("mm200_core_keys_unread") and mt == "mm200" and d.get("use_gated_attention", True):
        f.add("out_gate_post")                             # modeling_mm200.py:163-164, :229-232 (default True)
    if _r3("no_model_type_longcat") and mt == "longcat_flash":
        f.add("scmoe")                                     # every LongCat decoder layer is the ScMoE block
    if _r3("text_encoder_input_embeds") and isinstance(get("text_encoder_config"), dict):
        f.add("enc_input")                                 # (the outer Breeze config once the backbone is unwrapped)
    if (_r3("cross_attention_decoder_unflagged") and _truthy(d.get("add_cross_attention"))
            and _truthy(outer.get("is_encoder_decoder"))):
        f.add("cross_attn")
    if _r3("fake_quant_forward") and (d.get("_repo_id") or outer.get("_repo_id")) in FAKE_QUANT_REPOS:
        f.add("fake_quant")
    if _r3("gated_prenorm_lowrank") and mt in GATED_NORM_TYPES:
        f.add("gated_norm")
    if _r3("multimax_logits") and mt.startswith("ernie5_5") and _truthy(get("multimax")):
        f.add("multimax")                                  # config "multimax": "lm_head"
    if _r3("not_a_causal_lm") and _r3("nonlm_kept") and not _r3("nonlm_heads"):
        nl = _not_lm(outer or d, d)                        # round 3 (nonlm_kept): the non-vocab head / patch input
        if nl:                                             # (as features only with the head / input slots off)
            f.add(NONLM_FEAT[nl])
    if _r3("nonlm_heads") and head == "score" and _early_exits(d, outer):
        f.add("exit_heads")                                # (a non-default cutoff_layers saved in the config)
    # --- round 3, batch 2b (positions)
    qt = _q_temp(d, mt)
    if qt:
        f.add(qt)
    if _r3("fope_positions") and _fope(d):
        f.add("fope")
    if (_r3("content_warped_rope") and mt == "gptx2" and d.get("use_symplectic_rope") is True
            and "modeling_gptx2" in _auto_map_str(d, outer)):
        f.add("rope_warp")
    # --- round 3, batch 2c (windows and masks)
    if (_r3("swa_prefill_key_truncation") and mt in ("meditor", "meditor2") and _layer_count(d)
            and not all((i + 1) % (_num(d.get("global_attn_every_n_layers")) or 6) == 0
                        for i in range(_layer_count(d)))):
        f.add("swa_trunc")
    if _r3("doge_dynamic_mask") and mt == "doge":
        f.add("dyn_mask")
    if _r3("eos_document_mask") and ((mt == "ethoric" and d.get("doc_masking", True) is not False) or mt == "klm"):
        f.add("doc_mask")
    dros = get("d_ros_config")
    if (_r3("attn_output_saturation") and isinstance(dros, dict) and dros.get("enabled", True) is not False
            and "modeling_qwen3_eq" in _auto_map_str(d, outer)
            and float(dros.get("dry_wet", 0.70) if isinstance(dros.get("dry_wet", 0.70), (int, float)) else 0.70) > 0):
        f.add("attn_out_sat")
    # --- round 3, batch 2d (heads and MLA)
    if _r3("vha_head_mix") and mt == "ernie5_5_moe" and d.get("use_vha_attention", True) is not False:
        f.add("vha")                                       # VHAPostMix built in both attention kinds, unconditionally
    if (_r3("diff_attn") and mt == "motif" and str(d.get("attention_cls", "basic")).lower() == "gdla"
            and _num(d.get("num_noise_heads")) > 0):
        f.add("diff_attn")
    if _r3("xsa_self_value_projection"):
        if ((mt in ("gpts2", "gptx2") and d.get("xsa_projection") is True) or mt == "grug_moe"
                or (mt == "cyrah" and d.get("use_xsa", True) is True)):
            f.add("xsa")
        elif mt == "limite" and d.get("xsa", True) is True and d.get("xsa_layers"):
            f.add("xsa_scaled")
    if _canon(d, outer):
        f.add("canon")
    if _r3("cca_conv_qk_attention") and mt in ("zaya1_vl", "zaya") and d.get("cca", True) is not False:
        f.add("cca")
    if _moa(d):
        f.add("moa")
        f.add("bias_out")                                  # the MoA output bias (modular_jetmoe.py:228, :281)
    if _r3("memory_attention") and (
            (mt == "cfrd" and str(d.get("cell_attention", "local")).lower() != "full")
            or (mt == "fabric" and str(d.get("architecture", "fabric")).lower() == "fabric")
            or (mt == "edgerock" and d.get("fixed_memory_swa_layers")
                and (isinstance(d.get("independent_memory"), dict)
                     or "independent" in str(d.get("fixed_memory_revision") or "").lower()))):
        f.add("mem_attn")
    if (_r3("mova_routed_values") and mt in ("k2_horizon", "jais_asr") and _num(d.get("mova_num_experts")) > 0
            and _num(d.get("num_experts")) > 0):
        f.add("mova")
    if _r3("mla_query_raw_latent_skip") and mt == "axk2" and d.get("kv_lora_rank") and _num(d.get("q_lora_rank")) > 0:
        f.add("mla_q_raw")
    if _shared_mla(d):
        f |= {"mla_qmix", "bias_qkv"}                      # (mean_v: a V bias on the softmax-weighted values, :147)
    f |= _loop_feats(d, mt)                                # round 3, batch 4a: the loop boundaries
    return f


def _truthy(v):
    """A config switch the code tests with `if x` (or a string compare): True, 1, a non-empty string other than
    "false" / "none" / "no" / "0" / "off" (verify/r3/r3util.truthy_flag)."""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "none", "no", "0", "off")
    return bool(v)


# round 3 (dsv4_layer_types_schema): the tf5 DeepseekV4Config schema (configuration_deepseek_v4.py:157-165, :250-282):
# layer_types (explicit) > the legacy compress_ratios ({0: sliding, 4: CSA, 128: HCA}) > the default (every layer
# compressed); mlp_layer_types (explicit) > ["hash_moe"] * num_hash_layers (legacy) > default_num_hash_layers = 3. The
# attention picks its compressor by layer_types[i] (modeling_deepseek_v4.py:777), the MoE its router by
# mlp_layer_types[i] == "hash_moe" (:1126). MemT subclasses the config (configuration_memt.py:6, :20).
DSV4_SCHEMA_TYPES = frozenset({"deepseek_v4", "memt_deepseek_v4"})


def _dsv4_schema(d, mt):
    """(any compressed attention layer, any hash-routed MoE layer) of a DeepSeek-V4-schema config."""
    n = _num(d.get("num_hidden_layers"))
    lt, mlt, cr = d.get("layer_types"), d.get("mlp_layer_types"), d.get("compress_ratios")
    if isinstance(lt, list) and lt:
        comp = any("compressed" in str(x) for x in lt[:n or None])
    elif cr is not None:
        comp = isinstance(cr, list) and any(_num(x) for x in cr)
    else:
        comp = n > 0
    if isinstance(mlt, list) and mlt:
        hashed = any(str(x) == "hash_moe" for x in mlt[:n or None])
    elif d.get("num_hash_layers") is not None:
        hashed = _num(d.get("num_hash_layers")) > 0
    else:
        hashed = n > 0
    return comp, hashed


# round 3 (fake_quant_forward): BitCPM-CANN *-unquantized repos, whose LinearQuantizer ternarises every projection's
# weight in the forward (modeling_minicpm.py:129-162; 1B / 3B modeling_llama.py:141-147); their configs are identical to
# the stock ones, so the repo id (iter_json_configs' "_repo_id") is the only key
FAKE_QUANT_REPOS = frozenset({"openbmb/BitCPM-CANN-0.5B-unquantized", "openbmb/BitCPM-CANN-1B-unquantized",
                              "openbmb/BitCPM-CANN-3B-unquantized", "openbmb/BitCPM-CANN-8B-unquantized"})
# round 3 (gated_prenorm_lowrank): a low-rank sigmoid gate on the normed sublayer input: A.X-K2 (gated_norm_rank,
# default 16; every input_layernorm and the sparse layers' post_attention_layernorm) and Grug-MoE (rank 128 fixed in code,
# GRUG_GATED_NORM_RANK; every pre-norm, the embedding norm and the final norm)
GATED_NORM_TYPES = frozenset({"axk2", "grug_moe"})


def _gnorm_dims(d):
    """(gnorm_rank, gnorm_sites) of a gated-norm config (0, "") elsewhere; sites "layers" or "layers+emb+final"."""
    mt = str(d.get("model_type") or "").lower()
    if not _r3("gated_prenorm_lowrank") or mt not in GATED_NORM_TYPES:
        return 0, ""
    if mt == "axk2":
        return (_num(d.get("gated_norm_rank")) or 16), "layers"
    return 128, "layers+emb+final"


def _tied(d, outer=None):
    """lm_head = E^T: an explicit tie key (tie_word_embeddings, LFM2 tie_embedding, OLMo weight_tying, OpenELM
    share_input_output_layers, normalised to tie_word_embeddings), else the config class's default. Round 3: tied in
    code whatever the config says (TIED_BY_CODE); a typeless MiniCPM config reads the minicpm class default
    (tie_default_unread); OPT with word_embed_proj_dim != hidden is NOT tied at width h (factorized_embedding_proj:
    the input table is E P_in^T, the head E P_out, not transposes)."""
    outer = outer or {}
    mt = str(d.get("model_type") or "").lower()
    if mt in TIED_BY_CODE:
        return True
    if _r3("factorized_embedding_proj") and _emb_rank(d):
        return False
    for k in ("tie_word_embeddings", "tie_embedding", "weight_tying"):
        v = d.get(k, outer.get(k))
        if v is not None:
            return bool(v)
    if _r3("tie_default_unread") and not mt:
        archs = set(d.get("architectures") or ()) | set(outer.get("architectures") or ())
        if "MiniCPMForCausalLM" in archs:                  # typeless MiniCPM configs (configuration_minicpm.py:128)
            return bool(_class_default("minicpm", "tie_word_embeddings"))
    if _r3("tie_default_unread") and mt == "minicpm" and not (
            set(d.get("architectures") or ()) | set(outer.get("architectures") or ())) & {"MiniCPMForCausalLM"}:
        return False                                       # (a minicpm-typed reranker class: not the LM config class)
    return bool(_cfg_value(d, "tie_word_embeddings", outer))


# round 3 (tie_default_unread): the lm_head IS the embedding in code whatever the config says: Inferno
# (modeling_inferno.py:111), Mima (modeling_mima.py:139), CFRD (model.py:668-671 F.linear(final_norm(x),
# token_embedding.weight); haru / haru_1.1 save tie_word_embeddings false)
TIED_BY_CODE = _R3Set((), add={"tie_default_unread": {"cfrd", "mima", "inferno"}})


# q / k RMS norms built unconditionally by the modeling code (transformers 5.18 modeling_*.py: self.q_norm /
# self.q_layernorm in the attention __init__ without a config switch; code-checked for the edge-case audit)
QKNORM_ALWAYS = {"gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text", "gemma4_unified_text",
                 "diffusion_gemma", "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe",
                 "qwen3_5_moe_text", "qwen3_vl", "qwen3_vl_text", "qwen3_vl_moe", "qwen3_vl_moe_text",
                 "qwen3_omni_moe", "qwen4_exp", "qwen4_exp_text", "olmo2", "olmo3", "olmoe", "olmo_hybrid",
                 "flex_olmo", "afmoe", "apertus", "dots1", "exaone4", "exaone_moe", "glm5_next", "glm5_next_text",
                 "hunyuan_v1_dense", "hunyuan_v1_moe", "hy_v3", "laguna", "lfm2", "lfm2_moe", "mellum",
                 "minimax_m2", "nanochat", "nanochat_gpt", "step3p7", "doge", "chameleon",
                 # renamed types that keep their base family's q / k norms (literature review, round 2): SDAR and
                 # dLLM A2D (Qwen3 bodies), GKA-primed HQwen3 (Qwen3 softmax layers), PLaMo-2 (per-head weighted
                 # RMSNorm), DiffusionGemma's text model (Gemma-4), DeepSeek-V4 (q_b_norm per head + kv_norm)
                 "sdar", "sdar_moe", "a2d-qwen3", "hybrid_qwen3", "plamo2", "plamo2vl", "diffusion_gemma_text",
                 "flint",                               # Flint remote code: Qwen3MoeRMSNorm q_norm / k_norm per head
                 "deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4"}
# round 3, batch 2a: per-head q / k norms built with no config switch (remote / tf5 code; the spec cites each file:line):
# LFM3 / LFM3-RD, PLaMo-3 (offset gains), Alice-AI, GPT-S3 / -X3, EdgeRock (Qwen3_5Attention), Cyrah, Nexus,
# BananaMind2-MoE, Bitsy, KLM, MemT (Qwen3_5MoeAttention), the Qwen3 bodies (Isaac, Hulu-Med-Qwen3, MonkeyOCRv2,
# Qwen3Loop), Gemma3-MoE, Kolibri-1 (Exaone4Attention), Beam, IQuest-Q1, ViuAI, Needle (zero-centred) -- "head"; Domyn-
# Edge (head_qfree), Muse-Glimmer / Grug-MoE (head_free_gain), Dory (ngpt) by QKNORM_VARIANT_BY_TYPE
# (qknorm_by_code_unlisted). dots3-note's MLA rope-key norm (mla_k_rope_norm). GLM-5-Next's softmax layers are MLA with
# only the latent norms (modeling_glm5_next.py:1128-1137, :1188-1193): no q / k norm (glm5_next_mla_no_head_qknorm).
QKNORM_HEAD_BY_CODE = {"alice_ai", "bananamind2_moe", "beam", "bitsy_parallel", "cyrah", "edgerock", "gemma3_moe",
                       "gpts3_hybrid", "gptx3", "hulumed_qwen3", "iquest_q1", "isaac", "klm", "kolibri1", "lfm3", "lfm3_rd",
                       "memt_qwen3_5_moe", "monkeyocrv2", "needle", "nexus", "plamo3", "qwen3loop", "viuai"}
QKNORM_OTHER_BY_CODE = {"domynedge": "head_qfree", "muse_glimmer": "head_free_gain",
                        "muse_glimmer_text": "head_free_gain", "grug_moe": "head_free_gain", "dory": "ngpt"}
# round 3, the zoo's params triage (zoo_kanana2_qknorm): Kanana-2-tiny builds q_norm / k_norm RMSNorm(head_dim) on
# every attention with no config switch (modeling_kanana2_tiny.py:230-231, applied per head at :247-248; the checkpoint
# holds q_norm / k_norm [128] in every layer) -- "head" (the per-head weighted norm; its order against RoPE is the
# zoo's reading of the remote file, not re-read here)
QKNORM_ALWAYS = _R3Set(QKNORM_ALWAYS, add={"qknorm_by_code_unlisted": QKNORM_HEAD_BY_CODE | set(QKNORM_OTHER_BY_CODE),
                                           "mla_k_rope_norm": {"dots3_note"},
                                           "zoo_kanana2_qknorm": {"kanana2_tiny"}},
                       drop={"glm5_next_mla_no_head_qknorm": {"glm5_next", "glm5_next_text"}})
QKNORM_TYPES = QKNORM_ALWAYS
# a use_qk_norm key that does NOT put a q / k norm on the softmax layers: Ling / Ring 2.5 - 3.x apply it inside
# their linear-attention layers only (modeling_bailing_moe_v2_5), the MLA layers have none
QKNORM_KEY_NOT_SOFTMAX = {"bailing_hybrid",
                          "minicpm_sala",    # SALA: qk_norm goes to LightningAttention only (MiniCPMAttention has none)
                          # MLA softmax layers with no q / k norm although the config sets use_qk_norm (remote code
                          # checked, round 2 pass 2): Sarvam-105B (sarvam_mla: only the latent norms), Ming-3.0-ASR
                          "sarvam_mla", "ming_v3_asr"}
QKNORM_KEY_NOT_SOFTMAX_ARCHS = {"BailingMoeV3ForCausalLM"}   # Edge0 (model_type ''): MLA layers, no q / k norm
# a q / k norm in code with no config key (remote code checked): gaggle (nanochat port: weight-free after RoPE)
QKNORM_BY_CODE = {"gaggle"}
# round 3 (nanochat_port_unlisted): pit (rope(F.rms_norm(q)), modeling_pit.py:114-115) and sn38-nanochrono (RoPE then
# a weight-free F.rms_norm, modeling_nanochrono.py:50-51) normalise q / k per head with no key, as gaggle
QKNORM_BY_CODE = _R3Set(QKNORM_BY_CODE, add={"nanochat_port_unlisted": {"pit", "sn38-nanochrono"}})
# the qk-norm VARIANT a family's code builds (QKNORM_VARIANTS; transformers 5 modeling_*.py and the remote
# modeling files, round 2 pass 2). A family not listed with a per-head key falls to "head" only when its code was
# checked (QKNORM_HEAD_CHECKED); otherwise it is "unverified:<type>".
QKNORM_VARIANT_BY_TYPE = {
    **{t: "proj" for t in ("olmo2", "olmo3", "olmoe", "flex_olmo", "olmo_hybrid", "minimax_m2", "minimax")},
    **{t: "head_free" for t in ("llama4", "llama4_text", "nanochat", "nanochat_gpt", "arloop", "arloop_baseline",
                                "gaggle")},
    **{t: "head_post" for t in ("hunyuan_v1_dense", "hunyuan_v1_moe", "dhara_ar")},
    **{t: "head_distinct" for t in ("plamo2", "plamo2vl")},
    **{t: "head_ln" for t in ("cohere", "stablelm")},
    "chameleon": "head_ln_bias", "persimmon": "ln_shared_bias", "ceno": "qkv_input",
    **{t: "dsv4" for t in ("deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4")},
    # renamed types whose code is not public keep their base family's variant (as pass 1 keeps the base family's
    # features): Apertus-1.5 (Apertus: per-head q_norm before RoPE), HunyuanImage-3.0 (Hunyuan-A13B's MoE body:
    # query_layernorm after RoPE)
    "apertus1p5_text": "head", "hunyuan_image_3_moe": "head_post",
}
# round 3 (nanochat_port_unlisted): the ports' weight-free per-head q / k norm (order vs RoPE is irrelevant without a
# gain): head_free, as nanochat_gpt / gaggle
QKNORM_VARIANT_BY_TYPE = _R3Dict(QKNORM_VARIANT_BY_TYPE, add={
    "nanochat_port_unlisted": {"pit": "head_free", "sn38-nanochrono": "head_free"},
    "qknorm_by_code_unlisted": dict(QKNORM_OTHER_BY_CODE),
    # round 3 (dsv41_q_head_norm_absent): V4.1 keeps only the weighted K = V kv_norm
    "dsv41_q_head_norm_absent": {"deepseek_v41": "dsv4_kv", "deepseek_v41_text": "dsv4_kv"},
    "mla_k_rope_norm": {"dots3_note": "mla_krope"}})
# round 3 (qknorm_fixed_gain): a weight-free per-head q / k norm followed by a config constant (the logit gain g over
# 1 / sqrt(d)): (key, code default, applied to both q and k: g = s^2, else q only: g = s). nanochat_gpt
# modeling_nanochat_gpt.py:227-230, gaggle modeling_gaggle.py:83, :105, sn38-nanochrono modeling_nanochrono.py:51-53,
# Grug-MoE skyrl_train_models_grug_moe.py:732-733 (default 1.5703...), Muse-Glimmer modeling_muse_glimmer.py:340
# (default 3.87). Every such key is read from the config by its code, so a host of this variant sets c freely (a scale).
QK_GAIN_KEYS = {"nanochat_gpt": ("qk_sharpen_scale", None, True), "gaggle": ("qk_scale", None, True),
                "sn38-nanochrono": ("attention_multiplier", None, True),
                "grug_moe": ("qk_mult", 1.5703274004183786, False),
                "muse_glimmer": ("qk_scale_factor", 3.87, False), "muse_glimmer_text": ("qk_scale_factor", 3.87, False)}


def _qk_gain(d, mt):
    """The logit gain g != 1 of a weight-free qk-norm followed by a config constant (QK_GAIN_KEYS), else None."""
    if mt not in QK_GAIN_KEYS:
        return None
    key, default, both = QK_GAIN_KEYS[mt]
    v = d.get(key)
    s = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default
    if s is None:
        return None
    g = s * s if both else s
    return g if abs(g - 1.0) > 1e-9 else None


QKNORM_HEAD_CHECKED = {
    "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_vl",
    "qwen3_vl_text", "qwen3_vl_moe", "qwen3_vl_moe_text", "qwen3_omni_moe", "qwen4_exp", "qwen4_exp_text",
    "gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text", "gemma4_unified_text",
    "diffusion_gemma", "diffusion_gemma_text", "lfm2", "lfm2_moe", "apertus", "afmoe", "laguna", "mellum",
    "glm4_moe", "glm5_next", "glm5_next_text", "exaone4", "exaone_moe", "dots1", "doge", "hy_v3", "step3p5",
    "step3p7", "bailing_moe", "llada2_moe", "param2moe", "sarvam_moe", "kohaku", "kotodama", "gptx2", "lowonmind",
    "deep", "mm200", "openelm", "flint", "sdar", "sdar_moe", "a2d-qwen3", "hybrid_qwen3"}
QKNORM_HEAD_CHECKED = _R3Set(QKNORM_HEAD_CHECKED, add={"qknorm_by_code_unlisted": QKNORM_HEAD_BY_CODE | {"ethoric"},
                                                       "zoo_kanana2_qknorm": {"kanana2_tiny"}},
                             drop={"glm5_next_mla_no_head_qknorm": {"glm5_next", "glm5_next_text"}})


def _qknorm_variant(d, mt):
    """The qk-norm variant of a config whose softmax layers have a q / k norm (QKNORM_VARIANTS)."""
    nt = str(d.get("qk_norm_type") or "").lower()
    if nt == "per_layer":
        return "proj"                                       # MiniMax-M2 / -M3: one norm over the projection
    if mt == "bolmo":
        return "head" if d.get("use_head_qk_norm") else "proj"
    if mt in ("molmo", "hf_olmo", "olmo"):                  # attention_layer_norm over the whole projection
        lnt = str(d.get("layer_norm_type") or "").lower()
        return "proj" if lnt == "rms" else "proj_ln"
    if _r3("qknorm_fixed_gain") and _qk_gain(d, mt):
        return "head_free_gain"                             # round 3: a config gain c != 1 on the weight-free norm
    if _shared_mla(d):
        return "q_head"                                     # round 3 (hybrid_mla_layers_unread): q norm only
    if mt in QKNORM_VARIANT_BY_TYPE:
        return QKNORM_VARIANT_BY_TYPE[mt]
    if nt == "per_head" or mt in QKNORM_HEAD_CHECKED:
        return "head"
    return "unverified:" + (mt or "unknown")

# sigmoid-scored routers fixed in code (no config key; code-checked). With a selection-only expert bias too, unless
# listed in ROUTER_NO_BIAS_TYPES: DeepseekV3TopkRouter / Glm4MoeTopkRouter / Dots1 (sigmoid + e_score_correction_bias,
# whatever scoring_func says); NemotronHTopkRouter (Nemotron-H / Puzzle MoE); Laguna; Solar-Open (2); Hy4; AFMoE;
# EXAONE-MoE; LFM2-MoE (its bias is the use_expert_bias key); Flint (sigmoid, no bias); Llama-4 (sigmoid of the
# top-1 logit, no bias)
SIGMOID_ROUTER_TYPES = {"deepseek_v3", "glm4_moe", "glm4_moe_lite", "kimi_k2", "dots1", "glm_moe_dsa",
                        "deepseek_v32", "nemotron_h", "nemotron_h_puzzle", "laguna", "solar_open", "solar_open2",
                        "hy_v4", "afmoe", "exaone_moe", "lfm2_moe", "flint", "llama4", "llama4_text"}
ROUTER_NO_BIAS_TYPES = {"lfm2_moe", "flint", "llama4", "llama4_text"}
# a selection-only expert bias fixed in code on a softmax router or without a key: ERNIE-4.5-MoE (moe_statics
# e_score_correction_bias on the softmax probabilities), Kimi-Linear (e_score_correction_bias)
ROUTER_BIAS_TYPES = {"ernie4_5_moe", "kimi_linear"}
# round 3 (no_model_type_longcat): LongCat selects on scores + e_score_correction_bias, weights by the softmax scores
# (modeling_longcat.py:133-152); its configs are typeless, read as longcat_flash by architecture (_lm_config)
ROUTER_BIAS_TYPES = _R3Set(ROUTER_BIAS_TYPES, add={"no_model_type_longcat": {"longcat_flash"},
                                                    # round 3, batch 3a (router_bias_unread): a score-domain selection
                                                    # bias in code -- Kohaku expert_bias (modeling_kohaku.py:141, :159-160;
                                                    # dense_first_key_unread's side note, applied here once), Beam
                                                    # (modeling_beam.py:355), Muse-Spark (modular_muse_spark.py:331)
                                                    "router_bias_unread": {"kohaku", "beam", "muse_spark",
                                                                           "muse_spark_text"}})
# round 3, batch 3a (router_score_unread): sigmoid scoring fixed in code with no key p5 reads -- Beam (BeamTopKRouter,
# modeling_beam.py:354), Sarvam-105B (modeling_sarvam_moe.py:274-275 scoring_func / topk_method hard-coded), Grug-MoE
# (skyrl_train_models_grug_moe.py:544-553), Muse-Spark (modular_muse_spark.py:320), Kolibri-1 (modular_kolibri1.py:137),
# Mark-SLM-V3 (model_type deepseek2, the only snapshot carrier: modeling_mark_v3.py:195-200). Membership also gives the
# selection bias (each has one; Grug / Kolibri in the logit domain: ROUTER_BIAS_LOGIT_TYPES)
SIGMOID_ROUTER_TYPES = _R3Set(SIGMOID_ROUTER_TYPES, add={"router_score_unread": {
    "beam", "sarvam_mla", "grug_moe", "muse_spark", "muse_spark_text", "kolibri1", "deepseek2"}})
# o_proj bias = config.attention_bias (transformers 4.57 modeling_*.py: LlamaAttention and the bodies built from it)
O_BIAS_FROM_ATTN_BIAS = {"llama", "llama4", "llama4_text", "apertus", "arcee", "bitnet", "chameleon", "cohere", "cohere2",
                         "deepseek_v2", "deepseek_v3", "diffllama", "doge", "dots1", "flex_olmo", "gemma", "gemma2",
                         "gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gpt_neox", "gpt_oss", "granite",
                         "granitemoe", "granitemoehybrid", "granitemoeshared", "hunyuan_v1_dense", "hunyuan_v1_moe",
                         "longcat_flash", "nemotron", "olmo", "olmo2", "olmo3", "olmoe", "phimoe", "qwen3", "qwen3_moe",
                         "qwen3_next", "qwen3_omni_moe", "qwen3_vl", "qwen3_vl_text", "qwen3_vl_moe",
                         "qwen3_vl_moe_text", "smollm3", "vaultgemma", "falcon_h1", "bamba", "mellum",
                         "gpt_oss_puzzle"}
# a norm on V fixed in code: DeepSeek-V4's kv_norm (K = V, so its k-norm is a v-norm too)
VNORM_TYPES = {"deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4"}
# a sigmoid-gated shared expert in code (Qwen2MoeSparseMoeBlock.shared_expert_gate and its descendants)
SHARED_GATE_TYPES = {"qwen2_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_omni_moe",
                     "qwen4_exp", "qwen4_exp_text"}     # Qwen4-exp: the same gate (modeling_qwen4_exp.py, round 2 pass 3)
# round 3, batch 3a (shared_gate_undetected): AliceAISparseMoEBlock.shared_expert_gate (modeling_alice_ai.py:558-579) and
# MemT-1.0-Nano's Qwen3_5MoeSparseMoeBlock (modeling_memt.py:270-279): sigmoid(shared_expert_gate(x)) * shared(x)
SHARED_GATE_TYPES = _R3Set(SHARED_GATE_TYPES, add={"shared_gate_undetected": {"alice_ai", "memt_qwen3_5_moe"}})
# biases fixed in code (no config switch; code-checked): q / k / v projections, and the output side (o_proj /
# MLP biases, affine LayerNorm beta)
BIAS_QKV_TYPES = {"qwen", "qwen2", "qwen2_moe", "qwen2_vl", "qwen2_vl_text", "qwen2_5_vl", "qwen2_5_vl_text",
                  "qwen2_audio", "gpt2", "gpt_bigcode", "opt", "bloom", "gpt_neox", "gpt_neox_japanese", "xglm",
                  "phi", "persimmon", "starcoder2", "stablelm",
                  "dream", "jet_nemotron",              # Qwen2-style q / k / v biases (Dream; Jet-Nemotron's attention)
                  "biogpt", "ctrl", "openai-gpt", "bunny-phi",
                  # round 2 pass 2 (remote code checked): Phi-3-small query_key_value (nn.Linear default bias),
                  # Phi-4-mini-flash Wqkv bias=True
                  "phi3small", "phi4flash"}
BIAS_OUT_TYPES = {"gpt2", "gpt_bigcode", "opt", "bloom", "gpt_neox", "gpt_neox_japanese", "gptj", "codegen",
                  "gpt_neo", "xglm", "phi", "persimmon", "starcoder2", "stablelm", "falcon", "nemotron",
                  "gpt_oss",                            # gpt-oss: expert gate_up / down biases
                  "phimoe",                             # nn.LayerNorm beta, lm_head_bias
                  "progen", "gpt_neox_nope", "bunny-phi",   # fc_in / fc_out, dense biases and LN beta
                  "biogpt", "ctrl", "openai-gpt",
                  # round 2 pass 2 (code checked): nn.LayerNorm beta (Orion, Crowd, StableLM-Epoch remote), Phi-3-small
                  # (LayerNorm beta, dense / MLP biases), Phi-4-mini-flash (out_proj bias, LayerNorm beta), the
                  # gpt-oss-Puzzle (GptOssDecoderLayer: o_proj and expert biases), RWKV-4 / 5 / 6 LayerNorm beta
                  "orion", "crowd", "stablelm_epoch", "phi3small", "phi4flash", "gpt_oss_puzzle",
                  "rwkv", "rwkv5", "rwkv6"}
# round 3, batch 1: the code facts p5's tables missed
#   bias_qkv_by_code_unlisted: types that run transformers' Qwen2Attention (q / k / v bias=True, o bias=False:
#     modeling_qwen2.py:189-192) or their own biased q / k / v with no config key -- ArkASR (Qwen2ForCausalLM,
#     modeling_arkasr.py:7 / :75; audio_key_read_as_lm's bias half, applied once here), Step-Audio-2 (Qwen2Model,
#     modeling_step_audio_2.py:317), FastVLM / nanoLLaVA / Hulu-Med / VideoLLaMA3 / Cambrian-S / SpatialLM (Qwen2Model
#     bodies), Meditor (modeling_meditor.py:131-134), Sundial (modeling_sundial.py:109-112; placed, nonlm_kept)
#   bias_qkv_table_wrong: GPT-NeoX-Japanese's query_key_value is bias=False (modeling_gpt_neox_japanese.py:169); only
#     its last layer's dense_bias and the LayerNorm betas are biases (bias_out stays)
BIAS_QKV_TYPES = _R3Set(BIAS_QKV_TYPES, add={
    "bias_qkv_by_code_unlisted": {"arkasr", "cambrian_qwen", "hulumed_qwen2", "llava-qwen2", "llava_qwen2", "meditor",
                                  "spatiallm_qwen", "step_audio_2", "sundial", "videollama3_qwen2"},
    # round 3, batch 4b (cfrd_film_norm_shift): CFRD's phase_rms_norm adds a per-depth shift 0.1 s_d after every
    # sublayer-input norm (haru_1.1/model.py:185-198, :530-558): a norm beta, i.e. a constant into q / k / v and the MLP
    # inputs (a biased consumer's bias; the target side of bias_qkv is Lean biased_proj_not_bias_free)
    "cfrd_film_norm_shift": {"cfrd"}},
    drop={"bias_qkv_table_wrong": {"gpt_neox_japanese"}})
#   bias_out_by_code_unlisted: RecurrentGemma's o_proj and MLP are bias=True unconditionally
#     (modeling_recurrent_gemma.py:229, :489-491), JetMoE adds a MoE output bias (modeling_jetmoe.py:235, :274), Nae1's
#     norms are affine nn.LayerNorm (nae1_model.py:203-204, :236: the beta)
#   norm_fallback_rmsnorm_wrong: the LayerNorm beta of the Parler-TTS / MusicGen decoders (and sundial, placed: nonlm_kept)
BIAS_OUT_TYPES = _R3Set(BIAS_OUT_TYPES, add={
    "bias_out_by_code_unlisted": {"recurrent_gemma", "jetmoe", "nae1"},
    "cfrd_film_norm_shift": {"cfrd"},                    # (the same shift into the SwiGLU w1 / w3 and the memory reads)
    "norm_fallback_rmsnorm_wrong": {"parler_tts_decoder", "musicgen", "musicgen_decoder", "musicgen_melody",
                                    "musicgen_melody_decoder", "sundial"}})
# post-sublayer norms built unconditionally (transformers 5.18 decoder layers: post_feedforward_layernorm /
# post_mlp_layernorm / post_self_attn_layernorm applied to the sublayer output; code-checked)
POST_NORM_TYPES = {"gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text",
                   "gemma4_unified_text", "diffusion_gemma", "olmo2", "olmo3", "olmo_hybrid",
                   "flex_olmo", "exaone4", "exaone4_5", "glm4", "afmoe", "axk1", "muse_glimmer",
                   "muse_glimmer_text", "muse_spark", "muse_spark_text",
                   "diffusion_gemma_text", "plamo2", "plamo2vl"}     # (PLaMo-2: pre / post mixer and MLP norms)
# round 3 (norm_placement_keys_unread): sandwich norms fixed in code with no key -- PLaMo-3 post_mixer_norm /
# post_mlp_norm (modeling_plamo.py:575-608), Ethoric attn_out_norm / mlp_out_norm (modeling_ethoric.py:120-160), Domyn-Edge
# (modeling_domynedge.py:284-306), Ouro input_layernorm_2 / post_attention_layernorm_2 (modeling_ouro.py:399-443), the
# Gemma-2 cost-wise reranker (Gemma2DecoderLayer), Beam (modular_beam.py:307-341), Kolibri-1 (Glm4DecoderLayer);
# Needle and IQuest-Q1 post-norm the attention output (FEATURES post_norm_attn beside)
POST_NORM_TYPES = _R3Set(POST_NORM_TYPES, add={"norm_placement_keys_unread": {
    "plamo3", "ethoric", "domynedge", "ouro", "cost_wise_gemma", "beam", "kolibri1", "needle", "iquest_q1"},
    # round 3, the zoo's params triage (zoo_gemma3moe_post_norm): Gemma3-MoE's decoder layer IS Gemma3DecoderLayer with
    # the MLP swapped for the experts (modeling_gemma3_moe.py:72-75), so it keeps the four norms input / post_attention
    # / pre_feedforward / post_feedforward (tf5 modeling_gemma3.py:353-356): a sandwich, as gemma3_text
    "zoo_gemma3moe_post_norm": {"gemma3_moe"}})
# post-ONLY layers (no input_layernorm: transformers 5 Olmo2 / Olmo3 / FlexOlmo / Exaone4 / Exaone4_5 decoder
# layers; OLMo-hybrid's attention layers -- its recurrent layers are pre-norm): FEATURES post_only, not post_norm
POST_ONLY_TYPES = {"olmo2", "olmo3", "flex_olmo", "olmo_hybrid", "exaone4", "exaone4_5"}
# an attention output gate fixed in code (no config key)
OUT_GATE_TYPES = {"qwen3_next", "afmoe", "laguna", "minicpm_sala", "step3p5"}
# round 3 (out_gate_activation): Laguna's gate follows its gating key (False = none: modeling_laguna.py:458-459), read in
# _out_gate_key, not fixed by type
OUT_GATE_TYPES = _R3Set(OUT_GATE_TYPES, drop={"out_gate_activation": {"laguna"}})


# The RECURRENT FAMILY of a model's SSM / linear / conv layers (the R kind): features rf_<family>, ('wall', 'free')
# -- a target's recurrent layers need a host whose recurrent layers carry every rf_ feature the target's do; a host's
# recurrent layers of any family are zeroed for a target without them (ZERO_EDGES). attn/ssm_cross.py orders the bare
# recurrences (linear <= RetNet <= Mamba-2 <= GLA / Mamba-1 <= gated DeltaNet), but the real layers are not the
# bare recurrences: gated DeltaNet ties its write to its delta strength (beta_t both), Mamba-2 / GDN / KDA put a
# gated RMSNorm on the output that Mamba-1 lacks and cannot switch off, RWKV adds token shifts and a bonus,
# LFM2 has no recurrence at all (a gated short conv) -- so no two real families are claimed to nest.
# Round 2, pass 4 (literature review, recurrent / diffusion): the families are split into the variants that do not
# nest, each read from the modeling code (transformers 5.18 / the remote files). A variant that DOES nest in another
# carries both features (its code plays the narrower variant by a weight or config setting; RF_NESTS):
#   mamba1          plain Mamba-1 (MambaMixer: no norm on dt / B / C)
#   mamba1_bcdt     Falcon-Mamba: WEIGHT-FREE RMS norms on dt (rank dims), B, C (FalconMambaWeightlessRMSNorm, its own
#                   mixer_rms_eps); a per-token rescaling no plain Mamba-1 forms, and it cannot be switched off
#   mamba1_bcdtw    Jamba: WEIGHTED RMS norms there (JambaRMSNorm, eps = rms_norm_eps) -- also carries mamba1_bcdt:
#                   weight 1 is the weight-free norm, and the eps is free on both sides (the norms read x_proj rows, so
#                   scaling those rows by t gives the effective eps eps / t^2 whatever the shared config eps is)
#   hymba           Hymba: weighted dt / B / C norms AND a parallel attention + SSM layer whose two outputs are
#                   RMS-normed and averaged (cannot be switched off), 128 meta tokens: only Hymba plays Hymba
#   mamba1_mh       Zamba-1: multi-head Mamba-1 (n_mamba_heads: per-head x_proj / dt_proj, separate B / C)
#   sambay          Phi-4-mini-flash (SambaY): Mamba-1 layers plus GMU layers that gate on an earlier SSM output
#   mamba2          Mamba-2 / SSD with the gated RMSNorm after the gate, grouped by n_groups (Nemotron-H / CENO /
#                   Zamba2 / Falcon-H1 with mamba_rms_norm; full width = 1 group: Bamba, Granite-4.0-H)
#   mamba2_nonorm   Falcon-H1 with mamba_rms_norm false: y * silu(z), no norm (a normed host cannot drop it, an
#                   unnormed one cannot add it)
#   mamba2_nbg      Falcon-H1 mamba_norm_before_gate true: norm(y) * silu(z) (none in the snapshot)
#   plamo2          PLaMo-2: conv on x only, weighted RMS norms on B / C / dt from the post-conv x, no gated norm
#   gdn             gated DeltaNet (Qwen3-Next / 3.5 / 3.8 / Qwen4-exp, AllGDN, PseudoBrain conversions)
#   gdn_negeig      OLMo-Hybrid linear_allow_neg_eigval: beta = 2 sigmoid(b) in (0, 2) on the write AND the erase;
#                   no affine map of the logit gives 2 sigmoid(b) = sigmoid(b') -- a wall both ways
#   gdn2            GPT-S3 GDN-2: an erase gate on the delta term's read key (S -= S (e * k) k^T), a sigmoid-gated
#                   norm with a biased gate; sigmoid(.) = 1 only as a limit, so it does not nest over GDN
#   gka             Amazon GKA (hybrid_qwen3): a Chebyshev ridge readout over the key covariance, not gated DeltaNet
#   kda             Kimi Delta Attention, softplus decay g = -exp(A) softplus(f + dt_bias) (every KDA code has it)
#   kda_lb          ... the lower-bounded decay g = lb sigmoid(exp(A) (f + dt_bias)) (GLM-5-Next linear_lower_bound,
#                   Kimi-K3 gate_lower_bound, Youtu kda_gate_lower_bound, Ling-3 kda_safe_gate): those codes take the
#                   bound from the config (None = the softplus form), so they carry kda too; Kimi-Linear-48B's
#                   modeling_kimi.py has only the softplus form
# Round 3, batch 4a adds:
#   gdn_sig         Qwen4-exp output_gate_type sigmoid: RMSNorm(o) w sigmoid(z) (gdn_out_gate_act; a wall vs gdn)
#   kda_lb_negeig   Solar-Open2 kda_allow_neg_eigval: the lower-bounded KDA with beta = 2 sigmoid(b) (no nest)
#   deltanet        fla DeltaNet: the ungated delta rule with a plain per-head output norm (fla_num_heads_unplaced)
#   (and AliceAI -> kda, Cyrah's dual-gate conv -> shortconv, GPT-X3 / KLM -> gdn2, Edge0 / Ming-3 -> Ling-3's KDA)
# The dt floor (Nemotron-H / Zamba2 hard-code time_step_limit = (time_step_min, inf); Bamba / Granite / Falcon-H1 read
# time_step_limit) is no variant: time_step_min / time_step_limit are config values on every Mamba-2 code here (a tiny
# positive time_step_min, below fp32's smallest softplus output, is no floor; log(0) would break the init only).
RECURRENT_FAMILY = {
    "mamba": "mamba1", "falcon_mamba": "mamba1_bcdt", "jamba": "mamba1_bcdtw", "zamba": "mamba1_mh", "hymba": "hymba",
    "phi4flash": "sambay",
    "mamba2": "mamba2", "nemotron_h": "mamba2", "nemotron_h_puzzle": "mamba2", "ceno": "mamba2",
    "granitemoehybrid": "mamba2", "granite_switch": "mamba2", "falcon_h1": "mamba2", "zamba2": "mamba2",
    "bamba": "mamba2", "plamo2": "plamo2", "plamo2vl": "plamo2",
    "qwen3_next": "gdn", "qwen3_5": "gdn", "qwen3_5_text": "gdn", "qwen3_5_moe": "gdn", "qwen3_5_moe_text": "gdn",
    "qwen4_exp": "gdn", "qwen4_exp_text": "gdn", "olmo_hybrid": "gdn", "allgdn": "gdn", "qwen3_5_gdn24": "gdn",
    "memt_qwen3_5_moe": "gdn", "hybrid_qwen3": "gka", "gpts3_hybrid": "gdn2",
    "kimi_linear": "kda", "glm5_next_text": "kda", "glm5_next": "kda", "youtu_vl_moe": "kda",
    "lfm2": "shortconv", "lfm2_moe": "shortconv", "lfm3": "shortconv", "lfm3_rd": "shortconv",
    "recurrent_gemma": "rglru", "recurrentgemma": "rglru",
    "minimax": "lightning", "minimax_m1": "lightning", "minimax_text_01": "lightning",
    "rwkv": "rwkv4", "rwkv5": "rwkv5", "rwkv6": "rwkv6", "rwkv7": "rwkv7", "gla": "gla", "retnet": "retnet",
    "jet_nemotron": "jet",
}
# --- round 3, batch 4a: recurrent families read from the code -----------------------------------------------------------
#   kda_classified_as_gdn   AliceAI's linear_attention layers are AliceAIKDA (modeling_alice_ai.py:275-315, :338-385:
#                           per-key-channel decay -exp(A_log) softplus(alpha + dt_bias), low-rank decay / output gate,
#                           the Kimi-Linear block), not the gated DeltaNet the linear_num_value_heads fallback gave
#   fla_num_heads_unplaced  fla DeltaNet (delta_net.py:75-168): the delta rule with NO decay gate, a plain per-head
#                           o_norm (use_gate false) -- its own family "deltanet" (GDN gates its norm and decays)
#   hybrid_layer_schedule_unread  GPT-X3's GPTX3GDN2 and KLM's GatedDeltaNet2 (the GatedDeltaNet-2 rule: an erase gate,
#                           a sigmoid-gated biased norm, modeling_gptx3.py:108-127, modeling_klm.py:205-230) are gdn2;
#                           Solar-Open2's SolarOpen2LinearAttention is KDA (modeling_solar_open2.py:261-401; its
#                           kda_allow_neg_eigval variant is kda_lb_negeig, _recurrent_family)
RECURRENT_FAMILY = _R3Dict(RECURRENT_FAMILY, add={
    "kda_classified_as_gdn": {"alice_ai": "kda"},
    "fla_num_heads_unplaced": {"delta_net": "deltanet"},
    "hybrid_layer_schedule_unread": {"gptx3": "gdn2", "klm": "gdn2", "solar_open2": "kda"}})
# a variant whose code also runs another variant by a weight / config setting carries both features
RF_NESTS = {"mamba1_bcdtw": ("mamba1_bcdt",), "kda_lb": ("kda",)}


def _bailing_v3_alias(d):
    """Round 3, batch 4a (hybrid_layer_schedule_unread): a BailingMoeV3 stack under another name -- Edge0 (no
    model_type, BailingMoeV3ForCausalLM; modeling_bailing_moe_v3.py:1720-1721) and Ming-3.0-asr's text model
    (ming_v3_asr; modeling_ming_v3_asr.py:1217-1221) -- read like bailing_hybrid's Ling-3 (MLA every
    layer_group_size-th layer and the tail, KDA elsewhere)."""
    if not _r3("hybrid_layer_schedule_unread"):
        return False
    mt = str(d.get("model_type") or "").lower()
    return mt == "ming_v3_asr" or (not mt and "BailingMoeV3ForCausalLM" in (d.get("architectures") or ()))


def _recurrent_family(d, mixer):
    if mixer == "softmax":
        return None
    mt = str(d.get("model_type") or "").lower()
    if mt == "bailing_hybrid":
        # Ling-3.x (BailingMoeV3): KDA with a lower-bounded gate; Ling / Ring 2.5 - 2.6 (BailingMoeV2_5): a
        # lightning-style linear attention with fixed per-head decay, q / k norms and RoPE inside the recurrence and
        # a group norm -- not MiniMax's lightning, its own family
        arch = " ".join(d.get("architectures") or [])
        return _kda_variant(d) if "V3" in arch else "bailing_lightning"
    if _bailing_v3_alias(d):
        return _kda_variant(d)      # round 3 (hybrid_layer_schedule_unread): Edge0 / Ming-3 run Ling-3's KDA
    if mt == "minicpm_sala":
        return "sala_lightning"     # fixed ALiBi slopes (not layer-scaled), q / k norm + RoPE inside, no SiLU
    if _r3("cyrah_shortconv_is_lfm2") and mt == "cyrah":
        # round 3: Cyrah's dual-gate short conv is LFM2's out_proj(C * conv(B * x)) with the same chunk order
        # (modeling_cyrah.py:250-265 vs tf5 modeling_lfm2.py:344-386); Cyrah's in_proj has no bias whatever
        # conv_bias says while LFM2's follows it, so a conv_bias config (none in the snapshot) keeps its own family,
        # as does the single-gate layout (out_proj(gate * conv(x)): no B gate)
        if d.get("conv_dual_gate") and not d.get("conv_bias"):
            return "shortconv"
        return "other_cyrah"
    fam = RECURRENT_FAMILY.get(mt)
    if fam is None:
        if d.get("pb_hybrid_layers") or d.get("linear_num_value_heads") or d.get("gdn_layers_per_attention"):
            fam = "gdn"                                     # PseudoBrain / Qwen-derived gated DeltaNet
        elif d.get("mamba_num_heads") or d.get("mamba_head_dim") or d.get("mamba_n_heads"):
            fam = "mamba2"
        elif d.get("mamba_expand") or d.get("mamba_d_state") or d.get("ssm_state_size"):
            fam = "mamba1"
        else:
            return "other_" + (mt or "unknown")             # an unlisted family: only its own kind
    if fam == "mamba2" and mt == "falcon_h1":
        if _cfg_value(d, "mamba_rms_norm") is False:
            return "mamba2_nonorm"                          # FalconH1Mixer: self.norm = None, y * silu(z)
        if _cfg_value(d, "mamba_norm_before_gate"):
            return "mamba2_nbg"
    if (_r3("gdn_out_gate_act") and fam == "gdn" and mt in ("qwen4_exp", "qwen4_exp_text")
            and str(d.get("output_gate_type") or d.get("hidden_act") or "silu").lower() == "sigmoid"):
        # round 3: Qwen4-exp's gated norm applies ACT2FN[output_gate_type or hidden_act] to the gate
        # (modeling_qwen4_exp.py:184-193, :498-499): RMSNorm(o) w sigmoid(z), not Qwen3-Next's fixed silu(z) =
        # z sigmoid(z) -- no W_z makes one the other (sigmoid is bounded, silu is not): a wall both ways (needs theorem
        # gated_norm_gate_act_distinct for the claim; the wall needs none)
        return "gdn_sig"
    if mt == "olmo_hybrid" and d.get("linear_allow_neg_eigval", True) is not False:
        return "gdn_negeig"                                 # OlmoHybridGatedDeltaNet: beta = 2 sigmoid(b)
    if fam == "kda":
        if (_r3("hybrid_layer_schedule_unread") and mt == "solar_open2"
                and d.get("kda_gate_lower_bound", -5.0) is not None):
            # round 3: Solar-Open2's gate is always lower-bounded (getattr(config, 'kda_gate_lower_bound', -5.0) into
            # fused_kda_gate, modeling_solar_open2.py:278, :399), and allow_neg_eigval doubles beta (2 sigmoid(b) in
            # (0, 2), :279, :401): the gdn_negeig precedent, its own family (no RF_NESTS entry: no affine map of the
            # logit gives 2 sigmoid(b) = sigmoid(b'))
            return "kda_lb_negeig" if d.get("kda_allow_neg_eigval") is True else "kda_lb"
        return _kda_variant(d)
    return fam


def _kda_lower_bound(d):
    """The KDA decay's lower bound the code reads from the config (None = the softplus form): Kimi-K3 / GLM-5-Next
    linear_attn_config.gate_lower_bound, GLM-5-Next linear_lower_bound (class default -5.0), Youtu
    kda_gate_lower_bound, Ling-3 kda_lower_bound under kda_safe_gate."""
    mt = str(d.get("model_type") or "").lower()
    la = d.get("linear_attn_config") if isinstance(d.get("linear_attn_config"), dict) else {}
    if la.get("gate_lower_bound") is not None:
        return la["gate_lower_bound"]
    for k in ("linear_lower_bound", "kda_gate_lower_bound"):
        if d.get(k) is not None:
            return d[k]
    if d.get("kda_safe_gate"):
        return d.get("kda_lower_bound", -5.0)
    if mt.startswith("glm5_next") and "linear_lower_bound" not in d and la.get("safe_gate", True):
        return -5.0                                         # Glm5NextTextConfig: linear_lower_bound = -5.0
    return None


def _kda_variant(d):
    return "kda_lb" if _kda_lower_bound(d) is not None else "kda"


def _rf_feats(d, mixer):
    """The rf_ features of a config's recurrent layers: its family and the variants it nests (RF_NESTS)."""
    fam = _recurrent_family(d, mixer)
    if fam is None:
        return set()
    return {"rf_" + f for f in (fam,) + RF_NESTS.get(fam, ())}


# router features the MoE rules (moe_why on dims moe_sig) price between real points; FEATURES lists them as free
MOE_PRICED_FEATS = frozenset({"router_sigmoid", "router_sqrtsoftplus", "router_group"})


def feats_ok(t, h, strict=False, near_gap=False, host_off=frozenset()):
    """Target feature set t (a '+'-joined string) hosted by host feature set h: every target feature the
    host lacks is 'free' on the target side, every host feature the target lacks is 'free' on the host side
    ('limit' counts too, except under strict = the exact slice). host_off (round 2, pass 5): host features this
    host switches off by a lever (the eps lever's post-norms / v-norm), free on the host side."""
    ok_rule = {"free"} if strict else {"free", "limit"}
    # (round 3, qwen4_ple_reads_ngrams / ngram_memory_unread: an unregistered engram_* key is an n-gram memory, a wall
    # on the target side and switched off by its read projection on the host side, as rf_)
    rule = lambda x: FEATURES.get(x) or (("wall", "free") if x.startswith("rf_") or (
        x.startswith("engram_") and (_r3("qwen4_ple_reads_ngrams") or _r3("ngram_memory_unread")))
        else ("free", "free"))
    T = set(filter(None, (t or "").split("+")))
    if near_gap:                             # mode="mixgap": softmax plays recurrent layers, any family
        T = {x for x in T if not x.startswith("rf_")}
    if (h or "").startswith("*"):            # a host whose features are SWITCHABLE branches (the universal host;
        H = set(filter(None, h[1:].split("+")))   # 'rf_*' = every recurrent family: its gated-delta block)
        # (the universal host has no moe_sig: its router branches are its features, so a router feature it lacks --
        # MOE_PRICED_FEATS, priced by moe_why between real points -- is a wall here)
        return all((rule(x)[0] in ok_rule and x not in MOE_PRICED_FEATS) or any(p in H for p in _feat_parents(x, strict))
                   for x in T - H if not (x.startswith("rf_") and "rf_*" in H))
    H = set(filter(None, (h or "").split("+")))
    # round 3 (FEAT_NEST): a target feature the host lacks may be carried by a nesting parent the host has; that parent
    # is spent on it (not switched off)
    lack = T - H
    nest = {x: [p for p in _feat_parents(x, strict) if p in H] for x in lack}
    spent = {p for ps in nest.values() for p in ps}
    return (all(rule(x)[0] in ok_rule or nest[x] for x in lack)
            and all(rule(x)[1] in ok_rule or x in host_off or x in spent for x in H - T))


# --- classify one config into slot values -------------------------------------

def classify(d):
    """d is a plain config dict. Returns slot dict, or None (skipped/non-LM). Besides the SLOTS the dict
    carries "feats": the FEATURES the forward pass uses ('+'-joined, '' for none)."""
    outer = d
    d = _lm_config(d)
    if not isinstance(d, dict):
        return None
    mt = str(d.get("model_type") or "").lower()            # model types match case-insensitively ("Dream")
    if skipped(outer):
        return None                                        # non-decoder (outer or inner type) / drafter / assistant
    # sequence mixer -- place SSM/hybrid instead of dropping them
    if mt in SSM_TYPES:
        mixer = "ssm"
    elif mt in LINEAR_TYPES:
        mixer = "linear"
    elif mt in HYBRID_TYPES and not (isinstance(d.get("attn_type_list"), list) and d["attn_type_list"]
                                     and all(x == 1 for x in d["attn_type_list"])):
        mixer = "hybrid"                                   # (MiniMax-M2: attn_type_list all 1 = softmax only)
    elif (any(k in d for k in ("mamba_num_heads", "ssm_state_size", "mamba_d_state",
                               "mamba_expand", "mamba_head_dim", "linear_num_value_heads",
                               "linear_attention_pattern", "gdn_layers_per_attention"))
          or any(isinstance(d.get(k), list) and any(_rec_layer(x) for x in d[k])
                 for k in ("layers_block_type", "layer_types"))):
        # structural fallback: a brand-new model_type carrying mamba/SSM or linear-attention
        # machinery is a recurrent mixer, not softmax -- hybrid if it ALSO keeps full-attention
        # layers (e.g. Qwen3-Next's 3:1 linear:full, Nemotron-H's mamba+attn), else pure SSM.
        # Recurrent layer names include LFM's gated short 'conv', Jet-Nemotron's 'jet', 'GKA', ...
        lbt = [str(x).lower() for x in (d.get("layers_block_type") or d.get("layer_types")
                                        or d.get("hybrid_override_pattern") or [])]
        # any softmax-attention layer name ('qwen_sparse_attention', 'agnes_global_attention',
        # 'sliding_attention', ...) that is not itself a linear / delta / SSM recurrence
        has_full = any(x in ("attention", "full_attention", "attn", "*", "t", "a") or "full_attn" in x
                       or (("attention" in x or "attn" in x)
                           and not any(r in x for r in ("linear", "delta", "mamba", "ssm", "gdn")))
                       for x in lbt)
        if not lbt:            # no per-layer list: the softmax layers given by index / period
            lap = d.get("linear_attention_pattern")            # Youtu: 1 = linear, 0 = full layer
            n_l = _layer_count(d)
            hyb = [v for k, v in d.items() if k.endswith("hybrid_layers") and isinstance(v, list)]
            has_full = bool(d.get("full_attention_layers") or d.get("full_attention_interval")
                            or d.get("full_attention_idx") or d.get("mla_layers")
                            or d.get("gdn_layers_per_attention")       # GPT-S3: k GDN per softmax
                            or (isinstance(lap, list) and 0 in lap)
                            # PseudoBrain pb_hybrid_layers: a strict subset of the layers converted
                            or any(0 < len(v) < n_l for v in hyb))
        mixer = "hybrid" if has_full else "ssm"
    else:
        mixer = "softmax"
    if mixer == "hybrid" and _all_attention_list(d):
        mixer = "softmax"      # a hybrid CLASS run with no recurrent layer (Granite-4.0 micro / 1b / 350m:
        #                        GraniteMoeHybrid with every layer 'attention'; Qwen3.5 with all full_attention)
    nh = _num(d.get("num_attention_heads") or d.get("n_head")) or None    # gpt2/bloom use n_head (OpenELM: per layer)
    if nh is None and mixer == "softmax":
        return None                                        # a softmax model needs heads
    nkv = _num(d.get("num_key_value_heads") or d.get("num_kv_heads") or d.get("n_head_kv")
               or d.get("num_attention_groups")) or None   # Step3 MFA / Step-3.5: KV groups
    if not nkv and not d.get("multi_query"):
        nkv = _nkv_alias(d) or None                        # round 3 (n_kv_heads_key_unread): Nae1 n_kv_heads
    if not nkv and isinstance(d.get("num_key_value_heads_per_layer"), list):
        nkv = _num(d["num_key_value_heads_per_layer"]) or None  # DeciLM: per-layer GQA (widest)
    if not nkv and mt == "nemotron-nas" and isinstance(d.get("block_configs"), list) and nh:
        groups = [b.get("attention", {}).get("n_heads_in_group") for b in d["block_configs"]]
        groups = [g for g in groups if g]
        nkv = nh // min(groups) if groups else nkv             # per-block GQA (widest cache)

    # a "modern decoder" heuristic so a brand-new / unrecognized model_type still
    # places from its structure alone (rope-ish position, MoE experts, or a gated
    # family) instead of being dropped. Only used as the final norm/pos/act fallback.
    attn_cfg = d.get("attn_config") or {}
    rope_like = bool(d.get("rope_theta") or d.get("rope_parameters") or d.get("rope_scaling")
                     or d.get("rotary_dim") or d.get("rotary_pct") or d.get("partial_rotary_factor")
                     or d.get("swa_rope_theta") or attn_cfg.get("rope_theta")
                     or d.get("position_embedding_type") == "rotary"
                     # round 3, the zoo's params triage (zoo_phi3small_rope): Phi-3-small's theta key is
                     # rope_embedding_base (modeling_phi3_small.py's RotaryEmbedding); the -8k config has no other
                     # rope key and was left unplaced (pos None)
                     or (_r3("zoo_phi3small_rope") and d.get("rope_embedding_base")))
    n_exp_like = bool(d.get("num_experts") or d.get("n_routed_experts") or d.get("num_local_experts")
                      or d.get("moe_num_experts") or (d.get("ffn_config") or {}).get("moe_num_experts")
                      or d.get("num_routed_experts") or _block_experts(d))
    modern = rope_like or n_exp_like or mt in GATED

    # norm  (config keys first, family fallback for hardcoded ones)
    if _stock_arch(d) == "StableLmForCausalLM":
        norm = "LayerNorm"                                 # nn.LayerNorm whatever eps keys it carries
    elif mt in NORM_BY_CODE:
        norm = NORM_BY_CODE[mt]                            # what the modeling code builds
    elif mt == "molmo" and d.get("layer_norm_type"):
        norm = "RMSNorm" if d["layer_norm_type"] == "rms" else "LayerNorm"   # LayerNormBase.build
    elif "rms_norm_eps" in d or mt in NEMOTRON_H_TYPES or (mt == "chatglm" and d.get("rmsnorm")):
        norm = "RMSNorm"
    elif mt in RMS_LNEPS_TYPES or set(d.get("architectures") or ()) & RMS_LNEPS_ARCHS:
        norm = "RMSNorm"                                   # RMSNorm under a layer_norm_epsilon key
    elif "layer_norm_epsilon" in d or "layer_norm_eps" in d or mt in LN_FAMILIES:
        norm = "LayerNorm"
    else:
        norm = "RMSNorm" if (mt in GATED or modern) else None   # modern gated LMs are RMSNorm

    # positional
    if _is_alibi(d):
        pos = "alibi"
    elif (_r3("rope_embeddings_flag_unread") and mt == "parler_tts_decoder"
          and not _truthy(d.get("rope_embeddings", False))):
        # round 3: ParlerTTSDecoder builds the fixed Musicgen sinusoid table added to the input embeddings unless
        # rope_embeddings (default False; configuration_parler_tts.py:92-94, modeling_parler_tts.py:1358-1368,
        # :1506-1510; the attention rotates q / k only `if self.rope_embeddings`); its rope_theta key is unused
        pos = "sinusoidal"
    elif mt in SINUSOIDAL_TYPES:
        pos = "sinusoidal"                                 # a FIXED sinusoid table (XGLM, CTRL): no edge yet
    elif mt in LEARNED_ABS:
        pos = "learned_abs"
    elif (_r3("mpt_remote_alibi_default_false") and mt == "mpt" and not d.get("alibi")
          and d.get("learned_pos_emb", True) is not False):
        pos = "learned_abs"                                # (remote MPT, alibi off by its default: _normalize)
    elif rope_like or mt in GATED or mt in ("gpt_neox", "falcon") or modern:
        # (Falcon-7B / 40B carry no rope key: RoPE is the code path whenever alibi is off)
        pct = _rotary_factor(d, outer)
        rp = d.get("rope_parameters") if isinstance(d.get("rope_parameters"), dict) else {}
        # the factor nested in rope_parameters per layer type (Qwen3.5 / Gemma-4 / Laguna)
        nested = [v.get("partial_rotary_factor") for v in _rope_layer_types(rp).values() if v]
        if _r3("partial_rope_keys_unread") and mt == "step3p5" and isinstance(d.get("partial_rotary_factors"), list):
            nested += list(d["partial_rotary_factors"])    # (round 3: Step-3.5's per-layer factors, _rope_schedules)
        part = (pct is not None and pct < 1.0) or any(f is not None and f < 1.0 for f in nested)
        # a decoupled rotary slice of a wider head without an MLA latent (DeepSeek-V4: 64 of 512)
        qr, hd = _num(d.get("qk_rope_head_dim")), _num(d.get("head_dim"))
        part = part or bool(qr and hd and qr < hd and not d.get("kv_lora_rank"))
        rdim = d.get("rotary_dim") if not (_r3("rotary_dim_ignored") and _rotary_dim_unread(d, outer)) else None
        pos = "partial_rope" if part or rdim else "rope"
        if pct == 0 and not rdim:
            pos = "nope"                                   # rotary_pct = partial_rotary_factor = 0: RoPE off
        if d.get("qk_rope_head_dim") == 0 and _num(d.get("qk_nope_head_dim")):
            pos = "nope"                                   # MLA with no decoupled RoPE key (GLM-5.3-Flash)
    else:
        pos = None

    if mixer not in ("ssm", "linear") and _nope_attention(d, mt):
        pos = "nope"                                       # attention with no rotary at all (_nope_attention)

    # KV sharing (pure SSM has no attention KV -> sentinel)
    if mixer in ("ssm", "linear"):
        kv = "none"
        if pos is None:
            pos = "nope"
    elif d.get("kv_lora_rank"):
        kv = "MLA"
    elif d.get("multi_query"):
        kv = "MQA"
    elif nkv is None or nkv == nh or _moa(d):
        kv = "MHA"           # (round 3, moa_routed_attention_heads: JetMoE's per-block attention, n_kv q on n_kv KV)
    elif nkv == 1:
        kv = "MQA"
    else:
        kv = "GQA"

    # FFN
    ffn_cfg = d.get("ffn_config") or {}
    n_exp = (d.get("num_experts") or d.get("n_routed_experts") or d.get("num_local_experts")
             or d.get("moe_num_experts") or ffn_cfg.get("moe_num_experts")
             or d.get("num_routed_experts") or max((e for e, _ in _block_experts(d)), default=None))
    if _hier_moe(d):                  # round 3 (hierarchical_moe_unread): Meditor-2 HMoE, D x S sub-experts + shared
        n_exp = _hier_moe(d)[0] * _hier_moe(d)[1]
    if _moe_disabled(d) or _num(n_exp) <= 1:
        n_exp = None                  # MoE fields are unused defaults; one expert is a dense MLP (Hymba, 1x22b)
    shared = _has_shared_expert(d)
    if n_exp:
        ffn = "moe_shared" if shared else "moe"
    elif mt in HADAMARD_FFN_TYPES:
        ffn = "hadamard"              # round 3, batch 3b (hadamard_mlp_unlisted): needle's HadamardMLP, no gate / up / down
    elif mt in FFN_NONE_TYPES:
        ffn = "none"                  # round 3 (mamba_ffn_none): a pure Mamba block has no FFN (was vanilla / glu)
    elif mt in VANILLA or mt in NEMOTRON_H_TYPES or mt in UNGATED_BY_CODE or d.get("mlp_use_swiglu") is False:
        ffn = "vanilla"
    elif mt in GATED:
        ffn = "glu"
    else:
        ffn = "glu"   # modern default; unknown vanilla is rare

    # activation: the string the family's MLP reads (raw_act: ACT_KEY_BY_CODE, else the config keys, else the
    # config class default); GeLU erf and tanh are distinct values
    a = raw_act(d)
    if "silu" in a or "swish" in a or a == "quick_gelu":
        act = "SiLU"                                       # quick_gelu = x sigmoid(1.702 x) = SiLU(1.702 x) / 1.702
    elif a == "gegelu":
        act = "swigluoai"                                  # Phi-3-small: quick_gelu(g) * (u + 1), clamped = swigluoai
    elif mt in GELU_FORM_BY_CODE and (not a or "gelu" in a):
        act = GELU_FORM_BY_CODE[mt]
    elif "gelu" in a:
        act = "GeLU_tanh" if (a in GELU_TANH_NAMES or "tanh" in a) else "GeLU"
    elif a in RELU2_NAMES:
        act = "relu2"                                      # squared ReLU: not a swish limit (ACT_FLOORS)
    elif "relu" in a:
        act = "ReLU"
    elif a:
        act = a
    else:
        act = "SiLU" if (mt in GATED or mixer != "softmax" or modern) else None
    if mt in ("gpt_oss", "gpt_oss_puzzle"):
        act = "swigluoai"                                  # GptOssExperts: (up + 1) g sigmoid(1.702 g), +-7 clamps, in
        #                                                    code whatever hidden_act says (round 2 pass 3: was SiLU)

    # --- small config slots ---
    # block: parallel vs sequential attn+FFN. use_parallel_residual absent = the class default (GPTNeoXConfig:
    # True; gpt-neox-20b and the pythia-v0 configs omit it); Falcon: parallel_attn (class default True) or
    # new_decoder_architecture (Falcon-RW-1B is sequential)
    if mt == "falcon":
        par = bool(d.get("new_decoder_architecture") or _cfg_value(d, "parallel_attn"))
    else:
        par = bool(_cfg_value(d, "use_parallel_residual") or d.get("parallel_attn") or attn_cfg.get("parallel_attn")
                   or mt in PARALLEL_BY_CODE)                      # persimmon is sequential
    block = "parallel" if par else "sequential"
    # window: local (sliding) vs full attention (respect an explicit disable flag; a window >= max_position_embeddings
    # never binds and is dropped by _normalize)
    sw = d.get("sliding_window") or attn_cfg.get("sliding_window")
    window = "local" if (sw and d.get("use_sliding_window", True)) else "full"
    if any(w for _, w in _block_experts(d)):
        window = "local"                                   # gpt-oss-Puzzle: per-block windows
    lt_w = d.get("layer_types")
    if window == "full" and (_num(d.get("sliding_window_size"))
                             or (isinstance(d.get("sliding_windows"), list) and any(_num(w) for w in d["sliding_windows"]))
                             or (isinstance(lt_w, list) and any("sliding" in str(x) for x in lt_w))):
        window = "local"                                   # dots3 sliding_window_size, K-EXAONE-2.0 sliding_windows,
        #                                                    or sliding layers named in layer_types
    # qk-norm: RMSNorm on Q,K inside attention
    # (query_pre_attn_scalar is only the softmax scale -- Gemma-2 has no q / k norm; Cohere-2 and DeciLM /
    # Nemotron-NAS neither: code-checked, edge-case audit)
    uqn = d.get("use_qk_norm")
    if uqn is None and _r3("config_default_unread"):     # round 3: the class default when absent (LLaDA2-MoE True)
        uqn = _CLASS_DEFAULT_FALLBACK["use_qk_norm"].get(mt)
    qknorm = ("yes" if (uqn or d.get("qk_layernorm") or d.get("qk_norm") is True or d.get("use_head_qk_norm")
              or d.get("attention_layer_norm") or d.get("qkv_layernorm") or d.get("qk_norm_type")
              or mt in QKNORM_TYPES) else "no")
    if d.get("use_qk_norm") is False or d.get("qk_norm") is False or d.get("query_key_norm") is False:
        qknorm = "no" if mt not in QKNORM_ALWAYS else qknorm
    if mt in QKNORM_KEY_NOT_SOFTMAX or set(d.get("architectures") or ()) & QKNORM_KEY_NOT_SOFTMAX_ARCHS:
        qknorm = "no"
    if mt in QKNORM_BY_CODE and qknorm == "no" and d.get("qk_norm") is not False:
        qknorm = "yes"                                     # a q / k norm built without a config key (gaggle)
    if (_r3("qknorm_by_code_unlisted") and mt == "ethoric"
            and str(d.get("qk_norm") or "").lower() == "per_head"):
        qknorm = "yes"                                     # round 3: Ethoric's string flag (attention.py:162-166)
    if _r3("qknorm_always_remote_lacks") and mt == "doge" and "modeling_doge" in _auto_map_str(d, outer):
        qknorm = "no"                                      # round 3: the remote DogeAttention has no q / k norm
    if qknorm == "yes":
        qknorm = _qknorm_variant(d, mt)
    # logit soft-capping: on the attention scores ("attn", with or without a final cap) or the output logits
    # only ("final"). An absent attn_logit_softcapping key is the class default (Gemma2Config: 50.0); an
    # explicit null is no cap. dhara applies its logit_softcap only under use_logit_softcap.
    attn_cap = d["attn_logit_softcapping"] if "attn_logit_softcapping" in d else _cfg_value(d, "attn_logit_softcapping")
    final_cap = (d.get("final_logit_softcapping") or d.get("logits_soft_cap")
                 or (d.get("logit_softcap") if d.get("use_logit_softcap", True) is not False else None))
    if not final_cap and "final_logit_softcapping" not in d and _r3("config_default_unread"):
        final_cap = _CLASS_DEFAULT_FALLBACK["final_logit_softcapping"].get(mt)   # round 3: Muse-Glimmer 20.0
    softcap ="attn" if _num(attn_cap) > 0 else ("final" if _num(final_cap) > 0 else "no")
    # attention sinks (gpt-oss, MiMo-V2's sliding layers, Flint, DeepSeek-V4, ...), cross-layer KV sharing
    # (CLA/YOCO), multi-token prediction (DeepSeek-V3), generation paradigm (autoregressive vs diffusion)
    sink = "yes" if (any(d.get(k) is True or (k == "attention_sinks" and d.get(k)) for k in _SINK_KEYS)
                     or mt in SINK_TYPES) else "no"
    if _r3("sink_key_query_dependent") and mt == "iquest_q1" and d.get("enable_sink_attention") is True:
        sink = "key"                                       # round 3: a learned zero-value sink KEY (SINK_ORDER)
    if _r3("code_ignores_window_and_sink"):               # round 3: nanowhale's code ignores window and sink
        ov = _code_slot_override(d, outer)                 # (CODE_SLOT_OVERRIDES; _normalize drops the window)
        window, sink = ov.get("window", window), ov.get("sink", sink)
    crosslayer_kv = "yes" if (d.get("cross_layer_kv") or d.get("num_kv_shared_layers") or mt in CLA_TYPES
                              or (d.get("use_cla") and _num(d.get("cla_share_factor")) > 1)    # Hunyuan CLA
                              or _kv_source_shared(d)) else "no"
    mtp = "yes" if (d.get("num_nextn_predict_layers") or d.get("num_mtp_layers") or d.get("mtp_depth")) else "no"
    gen = _gen(d, mt, outer)
    loop ="looped" if _looped(d) else "untied"            # weight-tied / recurrent depth
    if loop == "looped" and _r3("loop_depth_unread"):
        # round 3: looped only when the unrolled forward is deeper than the distinct layers (LFM3-RD at recurrent_loops
        # = num_motifs runs every layer once; CFRD haru_2 has as many cells as recurrences): those are untied stacks
        pl, el = _loop_dims(d)
        loop = "looped" if el > pl else "untied"

    if mixer != "softmax":                                 # SSM/hybrid may lack norm/pos keys
        norm = norm or "RMSNorm"
        pos = pos or ("nope" if mixer in ("ssm", "linear") else "rope")
    no_attn = _r3("attn_feats_without_attention") and mixer in ("ssm", "linear")
    if no_attn:
        # round 3 (attn_feats_without_attention): a stack with no softmax layer (every mixer recurrent: Qwen3.8-
        # Longbook's qwen3_5_gdn24 "No rotary module: all 24 mixers are recurrent", modeling_qwen35_gdn24.py:121; AllGDN)
        # never applies the attention keys it inherits (attn_output_gate, partial_rotary_factor, qk-norm, an attention
        # softcap, a window): position-free, no q / k norm, no attention-only FEATURES
        pos, qknorm, window = "nope", "no", "full"
        if softcap == "attn":
            softcap = "final" if _num(final_cap) > 0 else "no"
    if None in (norm, pos, act):
        return None
    feats = _feats(d, outer if outer is not d else None) | _rf_feats(d, mixer)
    if no_attn:
        feats = {f for f in feats if f not in ATTN_ONLY_FEATS and not f.startswith("out_gate")}
    head, inp, _ = _readout(d, outer if outer is not d else None)     # round 3 (nonlm_heads): the readout / input
    return dict(norm=norm, pos=pos, kv=kv, ffn=ffn, act=act,
                block=block, window=window, qknorm=qknorm, softcap=softcap,
                mixer=mixer, sink=sink, crosslayer_kv=crosslayer_kv, mtp=mtp, gen=gen, loop=loop,
                head=head, input=inp, feats="+".join(sorted(feats)))


# round 3 (attn_feats_without_attention): the FEATURES that only a softmax attention layer applies (a stack whose every
# mixer is recurrent drops them; the out_gate* forms too)
ATTN_ONLY_FEATS = frozenset({"out_gate", "nope_layers", "dsa", "kv_compress", "clip_qkv", "vnorm"})


def _auto_map_str(d, outer=None):
    """The auto_map class paths of the LM config and its wrapper, joined (which remote modeling file runs)."""
    outer = outer if isinstance(outer, dict) else {}
    return " ".join(str(v) for x in (outer, d) if isinstance(x.get("auto_map"), dict) for v in x["auto_map"].values())


def _gen(d, mt, outer=None):
    """Generation paradigm: 'ar' (causal), 'diffusion' (bidirectional denoising), or 'block' (block-causal:
    bidirectional inside a block, causal across blocks -- SDAR, or block_size + mask_token_id); round 3: 'prefix'
    (a prefix LM: bidirectional over the prompt, causal after it)."""
    outer = outer if isinstance(outer, dict) else {}
    am = " ".join(str(v) for x in (outer, d) if isinstance(x.get("auto_map"), dict) for v in x["auto_map"].values())
    # round 3 (bidirectional_flag_true): Gemma-1 / 2 / 3 and Qwen3 read a bool use_bidirectional_attention as
    # is_causal = not it (modeling_gemma.py:239, gemma2 :231, gemma3 :320 / :547; voyage-4-nano's
    # modeling_qwen3_bidirectional.py:30-78); a remote *Bidirectional* class; the stella / gte remote modeling_qwen.py
    # Qwen2Model / Qwen2ForCausalLM.forward default is_causal=False (:970, :1165; the causal mask only `if is_causal`)
    bidir = _r3("bidirectional_flag_true") and (
        (d.get("use_bidirectional_attention") is True and mt in ("gemma", "gemma2", "gemma3", "gemma3_text", "qwen3"))
        or "Bidirectional" in am
        or ("modeling_qwen.Qwen2" in am and "is_causal" not in d and "is_causal" not in outer))
    if (mt in DIFFUSION_TYPES or "llada" in mt or "diffusion" in mt or d.get("is_diffusion")
            # a bidirectional (non-causal) stack: embedders (gte-Qwen2), Gemma-3 / 4 use_bidirectional_attention='all',
            # DiffuRefill attention_mask 'bidirectional' -- a causal host cannot see the future
            or d.get("is_causal") is False or d.get("use_bidirectional_attention") == "all"
            or str(d.get("attention_mask") or "").lower() == "bidirectional" or bidir):
        return "diffusion"
    # round 3 (prefix_lm_mask): DIVE-Doc's PaliGemma-style _update_causal_mask zeroes the mask over the whole prefill
    # outside training (modeling_divedoc.py:273-279: every prompt token attends every other), causal for decoding; MPT
    # attn_config.prefix_lm (no snapshot config sets it)
    archs = set(str(a) for x in (outer, d) for a in (x.get("architectures") or ()))
    if _r3("prefix_lm_mask") and (
            str(outer.get("model_type") or "").lower() in ("divedoc", "paligemma")
            or archs & {"DIVEdoc", "PaliGemmaForConditionalGeneration"}
            or (mt == "mpt" and _truthy((d.get("attn_config") or {}).get("prefix_lm")))):
        return "prefix"
    if mt in BLOCK_DIFFUSION_TYPES or (d.get("block_size") and d.get("mask_token_id") is not None):
        return "block"
    return "ar"


def _mla_off_rope(d):
    """Round 3 (use_mla_false_rope_dims): a DeepSeek-V2-type config with use_mla false and no KV latent runs the MHA
    attention class (LlamaAttention + LlamaRotaryEmbedding over the whole head: deepseek-vl2-tiny modeling_deepseek.py
    :1222-1223, :1232, :38-39), so its qk_rope_head_dim (0 there) is unread."""
    return (_r3("use_mla_false_rope_dims") and str(d.get("model_type") or "").lower() == "deepseek_v2"
            and d.get("use_mla") is False and not d.get("kv_lora_rank"))


def _rotary_factor(d, outer=None):
    """The fraction of each head that rotates: rotary_pct / partial_rotary_factor (top level, else nested flat in
    rope_parameters), else the config class's default (Persimmon 0.5, GLM / GLM-4 0.5, GPT-NeoX 0.25, StableLM
    0.25, Phi 0.5, Nemotron 0.5; ChatGLM 0.5 by code), else None."""
    mt = str(d.get("model_type") or "").lower()
    r3p = _r3("partial_rope_keys_unread")
    for k in ("rotary_pct", "partial_rotary_factor"):
        if d.get(k) is not None:
            return float(d[k])
    if r3p and mt == "stablelm_epoch" and d.get("rope_pct") is not None:
        # round 3 (partial_rope_keys_unread): StableLM-Epoch's remote code rotates int(head_dim * rope_pct) dims
        # (modeling_stablelm_epoch.py:209, :235-237): rope_pct is its rotary_pct
        return float(d["rope_pct"])
    rp = d.get("rope_parameters") if isinstance(d.get("rope_parameters"), dict) else {}
    if rp.get("partial_rotary_factor") is not None:
        return float(rp["partial_rotary_factor"])
    if r3p and mt == "k2_horizon" and _num(d.get("rope_head_dim")):
        # round 3 (partial_rope_keys_unread): K2-Horizon / Jais-3 rotate rope_head_dim of each head and pass the rest
        # (modeling_k2_horizon.py:210, :267-277; its split / interleave is one column permutation of q and k alike)
        nh = _num(d.get("num_attention_heads"))
        hd = _num(d.get("head_dim")) or (_num(d.get("hidden_size")) // nh if nh else 0)
        r = _num(d["rope_head_dim"])
        if hd and 0 < r < hd:
            return r / hd
    rd_unread = _r3("rotary_dim_ignored") and _rotary_dim_unread(d, outer)
    if (d.get("rotary_dim") and not rd_unread) or (d.get("qk_rope_head_dim") is not None and not _mla_off_rope(d)):
        return None                                       # the rotated dims are given explicitly
    if mt == "mm200" and _r3("mm200_core_keys_unread"):
        # round 3: MM200 rotates rot_dim = max(2, int(head_dim * partial_rope_frac) // 2 * 2) dims (default 0.25,
        # modeling_mm200.py:65, :125-126, :141-142; equal to round(frac * head_dim) on the snapshot's 64-dim heads)
        return float(d.get("partial_rope_frac", 0.25))
    for k in ("rotary_pct", "partial_rotary_factor"):
        v = _class_default(mt, k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


# attention layers with NO rotary at all, fixed in code (no rotary call in the attention: transformers 4.57 / 5.18
# modeling files and the repos' remote code): Nemotron-H / Puzzle and CENO (NemotronH attention), Jamba, Zamba-1,
# Phi-4-mini-flash (SambaY)
NOPE_BY_CODE = {"nemotron_h", "nemotron_h_puzzle", "ceno", "jamba", "zamba", "phi4flash"}


def _nope_attention(d, mt):
    """The softmax layers apply no positional encoding: NOPE_BY_CODE; Granite-4.0-H position_embedding_type 'nope';
    Kimi-Linear / K3 / GLM-5-Next MLA mla_use_nope; OLMo-Hybrid rope_theta null (rotary_emb = None); MiniCPM-SALA
    attn_use_rope false; Solar-Open2 use_rope false. (Round 3, audio_key_read_as_lm: ArkASR's use_rope belongs to its
    audio encoder; its LM is the stock Qwen2ForCausalLM, RoPE always, modeling_arkasr.py:8 / :92.)"""
    use_rope_off = d.get("use_rope") is False and not (mt == "arkasr" and _r3("audio_key_read_as_lm"))
    return (mt in NOPE_BY_CODE or str(d.get("position_embedding_type") or "").lower() == "nope"
            or d.get("mla_use_nope") is True or use_rope_off or d.get("attn_use_rope") is False
            or (mt == "olmo_hybrid" and not d.get("rope_theta")
                and not (d.get("rope_parameters") or {}).get("rope_theta"))
            # round 3 (use_mem_rope_unread): Zamba2's shared attention rotates q / k only under use_mem_rope (class
            # default False; tf5 configuration_zamba2.py:48, :100, modeling_zamba2.py:316-318, :1146-1151, :1205-1206)
            or (_r3("use_mem_rope_unread") and mt == "zamba2" and d.get("use_mem_rope") is not True))


def _llama4_shared(d):
    """round 3 (shared_expert_code_fixed_unread): Llama4TextMoe ALWAYS builds shared_expert = Llama4TextMLP(config),
    width intermediate_size, no config switch (tf5 modeling_llama4.py:88-96, 163-174)."""
    return bool(_r3("shared_expert_code_fixed_unread") and str(d.get("model_type") or "").lower() in ("llama4", "llama4_text")
                and _num(d.get("num_local_experts")) > 1)


def _has_shared_expert(d):
    """An always-on shared expert: a shared count / width key (a template width with n_shared_experts = 0 is not one:
    CENO), Hunyuan's num_shared_expert / use_mixed_mlp_moe, or Gemma-4's dense MLP beside the experts."""
    if _llama4_shared(d):
        return True
    if "n_shared_experts" in d and d["n_shared_experts"] is not None and _num(d["n_shared_experts"]) == 0:
        return _dense_mlp_beside_moe(d)
    return bool(d.get("n_shared_experts") or d.get("shared_expert_intermediate_size")
                or d.get("share_expert_dim") or d.get("moe_num_shared_experts")
                or d.get("moe_shared_expert_intermediate_size")
                or d.get("shared_intermediate_size")           # Granite-4.0-H shared_mlp
                or (_num(d.get("num_shared_experts")) > 0)     # Kimi-Linear / AFMoE / EXAONE-MoE / Hy3 / Ling
                or _num(d.get("num_shared_expert")) > 0 or d.get("use_mixed_mlp_moe")   # Hunyuan-v1 MoE
                or _dense_mlp_beside_moe(d))                   # Gemma-4 MoE: dense MLP beside the experts


def le(A, B):
    """A <= B: every slot of A reaches B's value. Returns (ok, [(slot,cost)])."""
    diffs = []
    for slot, order in SLOTS:
        a, b = A[slot], B[slot]
        if a == b:
            continue
        if b not in REACH[slot].get(a, {a}):
            return False, None
        diffs.append((slot, edge_cost(order, a, b)))
    return (len(diffs) > 0), diffs


# --- driver -------------------------------------------------------------------

def iter_local_configs(root):
    names = sorted(dd.split("models--", 1)[1].replace("--", "/")
                   for dd in os.listdir(root) if dd.startswith("models--"))
    for n in names:
        try:
            cfg = AutoConfig.from_pretrained(n, local_files_only=True, trust_remote_code=False)  # never run Hub code
            yield n, cfg.to_dict()
        except Exception:
            continue


def iter_json_configs(path):
    import json
    with open(path) as f:
        for row in json.load(f):
            c = row["config"]
            # round 3 (fake_quant_forward): the repo id rides along under "_repo_id" (no config key shows that code
            # fact; _feats reads it for FAKE_QUANT_REPOS only)
            yield row["id"], ({**c, "_repo_id": row["id"]} if isinstance(c, dict) else c)


JUNK = ("tiny-random", "internal-testing", "-random-", "/tiny-random")
DIM0 = {"hidden": 0, "layers": 0, "heads": 0, "inter": 0, "experts": 0, "kv_cap": 0,
        "q_cap": 0, "qheads": 0, "ffn_active": 0, "ffn_total": 0,
        "param_layers": 0, "eff_layers": 0, "vocab": 0,
        "attn_eff": 0, "attn_param": 0, "full_eff": 0, "full_param": 0}
# (the per-layer-kind dims k*_eff / k*_param and the *_u capacities are point-level only: a class max of
# kind counts is not a model)

# Vocab/tokenizer as a capacity axis is OFF by default: the reductions all live in
# HIDDEN space, and the embedding table + LM head are a resizable front-end. When ON
# (--with-vocab), B must have vocab_B >= vocab_A to hold A's token embeddings/logits.
# (Tokenizer *identity* -- which token maps to which id -- is orthogonal to the
# architecture and deliberately NOT modeled either way.)
USE_VOCAB = False
# Opt-in (like USE_VOCAB): price the max-context DEMAND and the tight window<->depth<->context
# coupling. A local-window host emulates a wider-reach target only by ALSO spending the depth
# ceil((reach-1)/(w-1)) to grow its receptive field to the target's -- a combination that must
# be met together, not an alternative. Off by default so the validated frontier is untouched.
USE_CONTEXT = False


def _junky(name):
    return any(j in name.lower() for j in JUNK)


def _num(v):
    """Coerce a config value to a scalar int. Heterogeneous models store some
    fields (intermediate_size, num_key_value_heads, ...) as PER-LAYER LISTS."""
    if isinstance(v, (list, tuple)):
        nums = [x for x in v if isinstance(x, (int, float))]
        return int(max(nums)) if nums else 0
    return int(v) if isinstance(v, (int, float)) else 0


# Multimodal / wrapper configs (Qwen3.5-VL, Gemma-4, Mistral3, Llava, DiffusionGemma,
# ...) nest the real text-transformer under text_config / llm_config / ... . Unwrap to
# it so slots + dims come from the actual decoder, not the empty wrapper shell.
_LM_NEST_KEYS = ("text_config", "llm_config", "llm_cfg", "language_config", "thinker_config",
                 "language_model", "text_model", "decoder")
_LM_WRAP_KEYS = ("qwen_config", "backbone_configs")
_LM_DIM_KEYS = ("hidden_size", "n_embd", "n_embed", "d_model", "hidden_dim",
                "model_dim", "dim", "embed_dim")
# layer count can also be a per-layer TYPE LIST (hybrid mamba/attn stacks name each
# layer, Reformer names each attn variant) -- its LENGTH is the depth.
_LM_LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers",
                  "num_transformer_layers", "n_layers")
_LM_LAYER_LIST_KEYS = ("layers_block_type", "attn_layers", "layer_types")


def _layer_count(d):
    """Scalar depth, or the LENGTH of a per-layer type list (hybrid/Reformer)."""
    for k in _LM_LAYER_KEYS:
        if d.get(k):
            return _num(d[k])
    for k in _LM_LAYER_LIST_KEYS:
        v = d.get(k)
        if isinstance(v, (list, tuple)) and v:
            return len(v)
    return 0


def _has_lm_dims(d):
    return (isinstance(d, dict)
            and any(d.get(k) for k in _LM_DIM_KEYS)
            and _layer_count(d) > 0)


def _make_divisible(v, divisor):
    """OpenELM's make_divisible (min_value = divisor)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


def _normalize(d):
    """Family-specific config keys -> the standard keys classify / dims_of read, each read from the family's
    code (a COPY; d itself is returned when nothing applies). The literature review's dropped / mis-read families:
      gpt_neo     num_heads; attention_layers global / local with window_size -> layer_types + sliding_window
      mpt         n_heads; attn_config (alibi, attn_type MQA / GQA kv_n_heads, clip_qkv, qk_ln); expansion_ratio
      xglm        attention_heads
      llada / hf_olmo  OLMo-style keys (n_heads, n_kv_heads, mlp_hidden_size / mlp_ratio, activation_type, block_type,
                  layer_norm_type, rope / alibi, weight_tying, include_bias, attention_layer_norm, max_sequence_length)
      openelm     per-layer num_query_heads / num_kv_heads / ffn_multipliers lists, share_input_output_layers
      internlm    InternLM-1's nested rotary {base, type}
      chatglm     multi_query_group_num KV groups, kv_channels head dim
      gpt2        SantaCoder's attention_head_type "multiquery"
      falcon      KV heads: num_kv_heads under new_decoder_architecture, else 1 when multi_query (class default True)
      bamba       attn_rotary_emb rotated dims
      lfm2        block_auto_adjust_ff_dim (Lfm2MLP: 2/3, multiplier, round up to block_multiple_of; block_ff_dim alias)
      llama4      no_rope_layers default: NoPE every no_rope_layer_interval-th layer (default 4)
      afmoe / exaone4 / exaone_moe / cohere2   RoPE only on the sliding layers: the full layers are NoPE
      bailing_hybrid  layer_group_size: an MLA layer every G-th (and the tail), linear attention elsewhere
      minicpm_sala    mixer_types: minicpm4 softmax / lightning-attn linear layers
    and, for every family, a sliding window >= max_position_embeddings (it never binds) is no window."""
    if not isinstance(d, dict):
        return d
    mt = str(d.get("model_type") or "").lower()
    u = {}
    g = d.get
    if mt == "gpt_neo":
        u["num_attention_heads"] = g("num_heads")
        al = g("attention_layers")
        if isinstance(al, list) and al:
            u["layer_types"] = ["sliding_attention" if str(x) == "local" else "full_attention" for x in al]
            if any(str(x) == "local" for x in al):
                u["sliding_window"] = g("window_size") or 256
    elif mt == "mpt":
        ac = g("attn_config") or {}
        nh = g("n_heads")
        at = str(ac.get("attn_type") or "multihead_attention")
        # round 3 (mpt_remote_alibi_default_false): the remote llm-foundry config fills a missing attn_config key from
        # attn_config_defaults, alibi False (configuration_mpt.py:4, :90-97; modeling_mpt.py:41, :54-55: no alibi ->
        # a learned wpe table, learned_pos_emb default True); the native MptAttentionConfig defaults it True
        remote = _r3("mpt_remote_alibi_default_false") and bool(g("auto_map"))
        u.update(num_attention_heads=nh, alibi=ac.get("alibi", not remote),    # MptAttentionConfig default alibi
                 num_key_value_heads=(1 if at.startswith("multiquery") else
                                      (ac.get("kv_n_heads") or nh) if at.startswith("grouped") else nh),
                 intermediate_size=int((g("expansion_ratio") or 4) * (g("d_model") or 0)),
                 max_position_embeddings=g("max_seq_len"))
        if ac.get("clip_qkv"):
            u["clip_qkv"] = ac["clip_qkv"]
        if ac.get("qk_ln"):
            u["qk_layernorm"] = True
    elif mt == "xglm":
        u["num_attention_heads"] = g("attention_heads")
    elif mt in ("llada", "hf_olmo"):
        nh = g("n_heads")
        hid = g("mlp_hidden_size") or int((g("mlp_ratio") or 4) * (g("d_model") or 0))
        act = str(g("activation_type") or "swiglu").lower()
        u.update(num_attention_heads=nh,
                 num_key_value_heads=g("n_kv_heads") or (1 if g("multi_query_attention") else nh),
                 # SwiGLU chunks ff_proj in two (OLMo sequential block); the llama block has separate gate / up
                 intermediate_size=hid // 2 if act == "swiglu" else hid,
                 hidden_act="silu" if act in ("swiglu", "silu") else act,
                 max_position_embeddings=g("max_sequence_length"),
                 tie_word_embeddings=bool(g("weight_tying")))
        if g("rope") and not g("rope_theta"):
            u["rope_theta"] = 10000.0
        if str(g("layer_norm_type") or "").lower() == "rms" and "rms_norm_eps" not in d:
            u["rms_norm_eps"] = 1e-5
        if g("include_bias"):
            u.update(attention_bias=True, mlp_bias=True)
        if g("include_qkv_bias"):
            u["qkv_bias"] = True
        if g("attention_layer_norm"):
            u["qk_norm"] = True
    elif mt == "openelm":
        dm, div = g("model_dim") or 0, g("ffn_dim_divisor") or 256
        mults = g("ffn_multipliers")
        n = g("num_transformer_layers") or 0
        mults = mults if isinstance(mults, list) else [mults or 4.0] * n
        u.update(num_attention_heads=g("num_query_heads"), num_key_value_heads=g("num_kv_heads"),
                 intermediate_size=[int(_make_divisible(m * dm, div)) for m in mults],
                 tie_word_embeddings=bool(g("share_input_output_layers")),
                 rope_theta=g("rope_freq_constant") or 10000, max_position_embeddings=g("max_context_length"),
                 hidden_act=g("activation_fn_name"))
        if str(g("normalization_layer_name") or "").startswith("rms"):
            u["rms_norm_eps"] = g("rms_norm_eps") or 1e-6
        if g("normalize_qk_projections"):
            u["qk_norm"] = True
    elif mt == "internlm" and isinstance(g("rotary"), dict):
        u["rope_theta"] = g("rotary").get("base", 10000)
        if g("rotary").get("type"):
            u["rope_scaling"] = {"type": g("rotary")["type"]}
    elif mt == "chatglm":
        if g("multi_query_attention") and g("multi_query_group_num"):
            u["num_key_value_heads"] = g("multi_query_group_num")
        if g("kv_channels"):
            u["head_dim"] = g("kv_channels")
    elif mt == "gpt2" and str(g("attention_head_type") or "") == "multiquery":
        u.update(num_key_value_heads=1, multi_query=True)                     # SantaCoder (remote GPT2 MQA)
    elif mt == "falcon":
        nh = g("num_attention_heads")
        if g("new_decoder_architecture"):
            nkv = g("num_kv_heads") or nh
        elif _cfg_value(d, "multi_query"):
            nkv = 1
        else:
            nkv = nh
        u.update(num_key_value_heads=nkv, multi_query=(nkv == 1))
    elif mt == "bamba" and g("attn_rotary_emb"):
        u["rotary_dim"] = g("attn_rotary_emb")
    if _r3("head_dim_key_unread") and g("head_dim") is None:
        # round 3: the head width the attention code reads under another key (q / k / v / o, kv_shapes and the RoPE
        # tables all follow it): Nemotron-H / CENO attention_head_dim, else (Nemotron-H, the key absent) the class
        # default 128 (modeling_nemotron_h.py:845-848, configuration_nemotron_h.py:143 / :186); Zamba / Zamba2
        # attention_head_dim = 2 hidden // heads over the 2h-wide concat (configuration_zamba.py:99-100,
        # modeling_zamba.py:133-143; zamba2 :114-115 / :246-256); JetMoE kv_channels, default 128
        # (configuration_jetmoe.py:48 / :54, modeling_jetmoe.py:453-460, rotary dim = head_dim :100); PLaMo-2
        # hidden_size_per_head, default 128 (modeling_plamo2.py:1042-1053, :115)
        hid, nh0 = _num(g("hidden_size")), _num(g("num_attention_heads"))
        hd0 = 0
        if nh0:
            if mt in ("nemotron_h", "nemotron_h_puzzle", "ceno"):
                if g("attention_head_dim") is not None:
                    hd0 = _num(g("attention_head_dim"))
                elif mt != "ceno" and "attention_head_dim" not in d:
                    hd0 = 128
            elif mt in ("zamba", "zamba2"):
                hd0 = _num(g("attention_head_dim")) or (2 * hid // nh0)
            elif mt == "jetmoe":
                hd0 = _num(g("kv_channels")) or 128
            elif mt in ("plamo2", "plamo2vl"):
                hd0 = _num(g("hidden_size_per_head")) or 128
        if hd0 and hd0 != hid // nh0:
            u["head_dim"] = hd0
    if _r3("switch_cache_layers_counted") and mt == "granite_switch" and _num(g("num_adapters")) > 0:
        # round 3: with adapters the composer inflates num_hidden_layers by SWITCH_CACHE_LAYERS = 2 (a counting and
        # a memory slot); the model builds num_hidden_layers - 2 decoder layers (granite-switch config.py:25-31 / :482,
        # modeling_granite_switch.py:439-444). layer_types is all full_attention, so which two go does not matter
        n_sw = _num(g("num_hidden_layers"))
        if n_sw > 2:
            u["num_hidden_layers"] = n_sw - 2
            if isinstance(g("layer_types"), list) and len(g("layer_types")) == n_sw:
                u["layer_types"] = list(g("layer_types"))[2:]
    if mt in ("lfm2", "lfm2_moe") and _cfg_value(d, "block_auto_adjust_ff_dim"):
        inter = g("block_ff_dim") or g("intermediate_size")
        if isinstance(inter, int) and inter:
            inter = int(2 * inter / 3)
            mult = g("block_ffn_dim_multiplier") if "block_ffn_dim_multiplier" in d else _class_default(
                mt, "block_ffn_dim_multiplier")
            if mult is not None:
                mo = g("block_multiple_of") or _class_default(mt, "block_multiple_of") or 256
                inter = int(mult * inter)
                inter = mo * ((inter + mo - 1) // mo)
            u["intermediate_size"] = inter
            if _r3("lfm2_normalize_idempotent"):
                # round 3, batch 3b (round3_inputs E): the adjusted width is the built one, so the copy no longer asks
                # for the adjustment -- without this a config with no block_ff_dim alias (ThoxFlux-350M, the
                # LFM2.5-350M / 1.2B finetunes) shrank again on every pass (6656 -> 4608 -> 3072 -> 2048): classify /
                # _pass5_dims re-read an already normalised LM config
                u["block_auto_adjust_ff_dim"] = False
    if mt in ("llama4", "llama4_text") and not g("no_rope_layers"):
        n = _num(g("num_hidden_layers"))
        iv = g("no_rope_layer_interval") or _class_default("llama4_text", "no_rope_layer_interval") or 4
        if n:
            u["no_rope_layers"] = [int((i + 1) % iv != 0) for i in range(n)]
    lt = u.get("layer_types", g("layer_types"))
    # round 3 (config_default_unread): EXAONE-MoE's class default sliding_window = 4096 (configuration_exaone_moe.py:102;
    # only an explicit None zeroes the pattern, :128-129): K-EXAONE-2.0-750B omits it, so its full layers are NoPE
    # (RoPE only `if sliding_window is None or is_sliding`, modeling_exaone_moe.py:184-185) and its sliding layers keep
    # 4096 tokens (:202)
    exm_default = (_r3("config_default_unread") and mt == "exaone_moe" and "sliding_window" not in d
                   and isinstance(lt, list) and any("sliding" in str(x) for x in lt))
    if exm_default:
        u["sliding_window"] = 4096
    # round 3 (nope_layer_keys_unread): Kolibri-1 runs the Exaone-4 attention (RoPE only `if sliding_window is None or
    # is_sliding`, tfmain modeling_kolibri1.py:150-152, :184) with a positive-validated class default window 513
    # (modular_kolibri1.py:77, :100-101): its full layers are NoPE
    kol = _r3("nope_layer_keys_unread") and mt == "kolibri1"
    if (mt in ("afmoe", "cohere2", "exaone4", "exaone4_5", "exaone_moe") or kol) and isinstance(lt, list) \
            and not g("no_rope_layers") and any("sliding" in str(x) for x in lt) \
            and (g("sliding_window") or mt in ("afmoe", "cohere2") or exm_default or (kol and "sliding_window" not in d)):
        u["no_rope_layers"] = [int("sliding" in str(x)) for x in lt]        # 1 = RoPE (sliding), 0 = NoPE (full)
    if _r3("nope_layer_keys_unread") and not g("no_rope_layers"):
        n = _num(g("num_hidden_layers"))
        if mt in ("muse_glimmer", "muse_glimmer_text", "muse_spark", "muse_spark_text"):
            # round 3: Muse-Glimmer / Muse-Spark pass position_embeddings only `if config.layer_rope_theta[i]` (tf5
            # modeling_muse_glimmer.py:521-522, tfmain modeling_muse_spark.py:666-667); absent, the class default is 0
            # (NoPE) every 4th layer counted from the last (configuration_muse_glimmer.py:147, :161-165;
            # configuration_muse_spark.py:141, :175-178)
            lrt = g("layer_rope_theta")
            if isinstance(lrt, list) and lrt:
                u["no_rope_layers"] = [int(bool(x)) for x in lrt]
            elif n:
                u["no_rope_layers"] = [0 if (n - 1 - i) % 4 == 0 else 1 for i in range(n)]
        elif mt == "domynedge" and isinstance(g("position_layers"), list) and g("position_layers"):
            # round 3: Domyn-Edge rotates only where position_layers[i] == "rope" (modeling_domynedge.py:208, :242-244)
            u["no_rope_layers"] = [int(str(x) == "rope") for x in g("position_layers")]
    if _r3("nope_full_layers_unlisted") and mt == "grug_moe" and not g("no_rope_layers"):
        # round 3: Grug-MoE's long layers ((idx + 1) % 4 == 0 or the last: grug_long_layer_flags, skyrl_train_models_
        # grug_moe.py:50, :91-98, :979) skip RoPE (disable_long_rope forced True, :193, :286-287, :355; `if not
        # is_long: _apply_half_rope`, :730-731)
        n = _num(g("num_hidden_layers"))
        if n:
            u["no_rope_layers"] = [0 if ((i + 1) % 4 == 0 or i == n - 1) else 1 for i in range(n)]
    if (_r3("fla_num_heads_unplaced") and not g("num_attention_heads") and _num(g("num_heads"))
            and set(g("architectures") or ()) & FLA_ARCHS):
        # round 3: flash-linear-attention configs name heads num_heads / num_kv_heads (fla-hub transformer / delta_net /
        # gla; modeling_utils.py:32-41 get_attention_spec, modeling_delta_net.py:44-55); the transformer's MLP width
        # is fla's GatedMLP rule when intermediate_size is null (mlp.py:46-50; gla / delta_net read it through
        # _ffn_width_by_code, round 3 batch 3b)
        u.update(num_attention_heads=g("num_heads"), num_key_value_heads=g("num_kv_heads") or g("num_heads"))
        if g("intermediate_size") is None and "TransformerForCausalLM" in g("architectures"):
            hr = g("hidden_ratio") if isinstance(g("hidden_ratio"), (int, float)) else 4
            u["intermediate_size"] = 256 * -(-int(_num(g("hidden_size")) * (hr or 4) * 2 / 3) // 256)
    if _r3("hybrid_layer_schedule_unread"):
        # round 3: hybrid schedules p5 did not read (the rest of the stack recurrent): GPT-X3 softmax iff
        # (i + 1) % (gdn_layers_per_gqa + 1) == 0 (configuration_gptx3.py:48-49); KLM iff i % attention_every ==
        # attention_every - 1 (configuration_klm.py:55-56, default 4); Solar-Open2 GQA iff i in gqa_layers, else
        # (i + 1) % gqa_interval == 0 (modeling_solar_open2.py:779-783)
        n = _num(g("num_hidden_layers"))
        sched = None
        if mt == "gptx3" and _num(g("gdn_layers_per_gqa")):
            k = _num(g("gdn_layers_per_gqa"))
            sched = [(i + 1) % (k + 1) == 0 for i in range(n)]
        elif mt == "klm":
            e = _num(g("attention_every")) or 4
            sched = [i % e == e - 1 for i in range(n)]
        elif mt == "solar_open2":
            gl = g("gqa_layers")
            if isinstance(gl, list):
                sched = [i in {_num(x) for x in gl} for i in range(n)]
            elif _num(g("gqa_interval")):
                sched = [(i + 1) % _num(g("gqa_interval")) == 0 for i in range(n)]
        if sched and not all(sched) and not isinstance(g("layer_types"), list):
            u["layer_types"] = ["full_attention" if a else "linear_attention" for a in sched]
    if (mt == "bailing_hybrid" or _bailing_v3_alias(d)) and (g("layer_group_size") or mt == "ming_v3_asr"):
        # (round 3, hybrid_layer_schedule_unread: Edge0 / Ming-3 by the same rule; Ming's layer_group_size defaults
        # to 5, configuration_ming_v3_asr.py:72)
        n, G = _num(g("num_hidden_layers")), _num(g("layer_group_size")) or 5
        u["layer_types"] = ["full_attention" if ((i + 1) % G == 0 or i >= n // G * G) else "linear_attention"
                            for i in range(n)]
    if mt == "minicpm_sala" and isinstance(g("mixer_types"), list):
        u["layer_types"] = ["linear_attention" if "lightning" in str(x) else "full_attention" for x in g("mixer_types")]
    if _r3("sliding_window_keys_unread"):
        u.update(_sliding_layout(d, mt))
    if _r3("swa_window_plus_one") and mt in INCLUSIVE_WINDOW_TYPES and _num(g("sliding_window")) \
            and isinstance(g("layer_types"), list) and any("sliding" in str(x) for x in g("layer_types")):
        # round 3: sn38-nanochrono's _attend keeps key j iff row - col <= window (modeling_nanochrono.py:71-79), so a
        # sliding layer sees sliding_window + 1 keys (the HF band sees sliding_window)
        u["sliding_window"] = _num(g("sliding_window")) + 1
    if _r3("code_ignores_window_and_sink") and _code_slot_override(d).get("window") == "full":
        # round 3: nanowhale's remote modeling_deepseek_v4.py builds its own triu causal mask and never reads
        # config.sliding_window (nanowhale-100m-base/modeling_deepseek_v4.py:582-585): every layer is full causal
        u.update(sliding_window=None, use_sliding_window=False)
    drop = ()
    if _r3("v_head_dim_narrow") and mt == "needle" and _num(g("qk_head_dim")) and g("head_dim") is None:
        # round 3: Needle's q / k heads are qk_head_dim wide (cactus architecture.py:174-175; V value_head_dim)
        u["head_dim"] = _num(g("qk_head_dim"))
    if _r3("cyrah_attn_impl_unread") and mt == "cyrah" and str(g("attn_impl") or "mla").lower() == "gqa":
        # round 3: Cyrah's attn_impl "gqa" (modeling_cyrah.py:376-395, :432-454) is plain GQA with per-head q / k
        # norms and full RoPE (rotary_dim = nope + rope = head_dim, :632): no q / kv LoRA latent, no indexer (built
        # only after the gqa return, :413-415), and a window only on 'sliding_attention' layers (:529) -- none here
        drop = ("kv_lora_rank", "q_lora_rank", "index_topk", "index_n_heads", "index_head_dim")
        hd_c = _num(g("qk_nope_head_dim")) + _num(g("qk_rope_head_dim"))
        if hd_c:
            u["head_dim"] = hd_c
            if _num(g("rotary_dim")) >= hd_c:
                drop = drop + ("rotary_dim",)
        lt_c = u.get("layer_types", g("layer_types"))
        if not (isinstance(lt_c, list) and any(str(x) == "sliding_attention" for x in lt_c)):
            u.update(sliding_window=None, use_sliding_window=False)
    sm = _shared_mla(d)
    if sm:
        # round 3 (hybrid_mla_layers_unread): Sabi's shared pre-RoPE MLA on its full-attention layers (configuration_
        # sabi_mla.py:30-62, modeling_sabi_mla.py:88-148): k / v from a 512-wide latent with NO latent norm, a shared
        # 64-dim rotary key, per-head v 256, no q latent -- the MLA dims (the other features: _feats mla_qmix / bias_qkv)
        u.update(kv_lora_rank=_num(sm["kv_lora_rank"]), qk_rope_head_dim=_num(sm["qk_rope_head_dim"]),
                 qk_nope_head_dim=_num(sm["qk_nope_head_dim"]), v_head_dim=_num(sm["v_head_dim"]))
        drop = drop + ("q_lora_rank",)
    sw0 = u.get("sliding_window", g("sliding_window"))
    sw, mpe = _num(sw0), _num(u.get("max_position_embeddings") or g("max_position_embeddings"))
    if sw and mpe and sw >= mpe and not isinstance(sw0, list):
        u.update(sliding_window=None, use_sliding_window=False)
        lt = u.get("layer_types", g("layer_types"))
        if isinstance(lt, list):
            u["layer_types"] = [("full_attention" if str(x) == "sliding_attention" else x) for x in lt]
    u = {k: v for k, v in u.items() if v is not None or k == "sliding_window"}
    if drop:
        out = {**d, **u}
        for k in drop:
            out.pop(k, None)
        return out
    return {**d, **u} if u else d


# Sabi-Noema's DEFAULT_SHARED_MLA (configuration_sabi_mla.py:30-41), merged under the config's shared_mla block (:57-62)
_SABI_SHARED_MLA = {"enabled": True, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 384,
                    "v_head_dim": 256}


def _shared_mla(d):
    """Round 3 (hybrid_mla_layers_unread): the merged shared_mla block of a Sabi config (enabled, with listed layers),
    else None."""
    if not _r3("hybrid_mla_layers_unread"):
        return None
    sm = d.get("shared_mla")
    if not isinstance(sm, dict) or sm.get("enabled", True) is False:
        return None
    blk = {**_SABI_SHARED_MLA, **sm}
    return blk if [i for i in (blk.get("layers") or []) if isinstance(i, int)] else None


# --- round 3, batch 2c: windows and masks -------------------------------------------------------------------------
# swa_window_plus_one: the nanochat-style inclusive band (row - col <= window: window + 1 keys). gaggle / Motif, the
# same convention, are read in _sliding_layout (sliding_window_keys_unread).
INCLUSIVE_WINDOW_TYPES = _R3Set(add={"swa_window_plus_one": {"sn38-nanochrono"}})
# code_ignores_window_and_sink: a remote modeling file (the auto_map class path) whose forward ignores config slots the
# config names. nanowhale (HuggingFaceTB/nanowhale-100m / -base) runs its own modeling_deepseek_v4.py: "always create
# our own 4D mask" triu causal (modeling_deepseek_v4.py:582-585, never reads sliding_window), and its attn_sink
# parameter is built (:170-171) but "omitted in SDPA path" (:217-225: no sink logit). The transformers-code V4
# snapshots keep their window and sink. Only without compressed layers (compress_ratios all 0).
CODE_SLOT_OVERRIDES = _R3Dict(add={"code_ignores_window_and_sink": {
    "modeling_deepseek_v4.DeepseekV4ForCausalLM": {"window": "full", "sink": "no"}}})


def _code_slot_override(d, outer=None):
    """The slot values a remote modeling file forces over its config (CODE_SLOT_OVERRIDES), else {}."""
    am = _auto_map_str(d, outer)
    cr = d.get("compress_ratios")
    for path, slots in CODE_SLOT_OVERRIDES.items():
        if path in am and not (isinstance(cr, list) and any(_num(x) for x in cr)):
            return slots
    return {}


def _int_list(v):
    """A list of ints (a JSON string of one too), else []."""
    if isinstance(v, str):
        try:
            import json
            v = json.loads(v)
        except ValueError:
            return []
    return [int(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)] if isinstance(v, list) else []


def _sliding_layout(d, mt):
    """round 3 (sliding_window_keys_unread): the standard layer_types / sliding_window of the families whose window and
    layer schedule sit under their own keys, each from the family's code. 'sliding_window' is the number of keys a
    sliding layer sees (W + 1 for an inclusive band). Returns the keys to set ({} when nothing applies)."""
    g = d.get
    n = _layer_count(d)
    if not n:
        return {}
    sl = lambda kinds: ["sliding_attention" if k == "local" else ("full_attention" if k == "full" else k)
                        for k in kinds]
    if mt == "cost_wise_gemma" and _num(g("sliding_window")):
        # BAAI's CostWiseGemma builds transformers-4.42 Gemma2DecoderLayers (gemma_model.py:56, :258): is_sliding = not
        # layer_idx % 2 (4.42 modeling_gemma2.py:624), band tril(diagonal=-sliding_window) (:640-645). (A reranker:
        # placed, nonlm_kept. Its code USES the window: gemma_model.py never reads sliding_window itself -- its own
        # _update_causal_mask (:440-482) is the plain causal mask -- but every layer is the imported Gemma2DecoderLayer,
        # which bands the even layers at config.sliding_window = 4096 < max_position_embeddings 8192. NOT window-free:
        # the 4.42 layer that would make it a host lever is not on disk, see WINDOW_FREE_TYPES.)
        return {"layer_types": sl(["local" if i % 2 == 0 else "full" for i in range(n)])}
    if mt == "needle":
        # cactus architecture.py:469-475: the band pos_q - pos_k < sliding_window on every layer but global_layers
        # (config.py:36), kept in extras.global_attention_layers (needle2 lists none: all local, already read)
        gl = set(_int_list((g("extras") or {}).get("global_attention_layers") if isinstance(g("extras"), dict)
                           else None))
        return {"layer_types": sl(["full" if i in gl else "local" for i in range(n)])} if gl else {}
    if mt == "iquest_q1" and g("use_hybrid_layers", True) is not False and g("use_sliding_window", True) is not False:
        # IQuestQ1Config: layer_types = first + block * num_hybrid_layers_block + last (configs.py:72-99), per-layer
        # window model.py:70-90
        first = list(g("first_layers_types") or ["full_attention"])
        block = list(g("hybrid_layers_types_block") or ["full_attention"] + ["sliding_attention"] * 3)
        last = list(g("last_layers_types") or ["full_attention"] * 3)
        lt = first + block * (_num(g("num_hybrid_layers_block")) or 21) + last
        return {"layer_types": [str(x) for x in lt]} if len(lt) == n else {}
    if mt == "motif" and g("use_sliding_window") and _num(g("sliding_window")):
        # Motif-3: effective_window = sliding_window + 1; 'interleave': SWA iff layer_idx % period != 0
        # (modeling_motif.py:568-577), flash only (:1297-1307)
        pat, per = str(g("sliding_window_pattern") or "interleave"), _num(g("sliding_window_period")) or 2
        if pat not in ("all", "interleave"):
            return {}
        kinds = ["local" if (pat == "all" or i % per != 0) else "full" for i in range(n)]
        return {"layer_types": sl(kinds), "sliding_window": _num(g("sliding_window")) + 1}
    if mt in ("meditor", "meditor2"):
        # is_global_layer = (layer_idx + 1) % global_attn_every_n_layers == 0, else sliding_window_size
        # (modeling_meditor.py:122-123, modeling_meditor2.py:130-131; their prefill: swa_prefill_key_truncation)
        ge = _num(g("global_attn_every_n_layers")) or 6
        return {"layer_types": sl(["full" if (i + 1) % ge == 0 else "local" for i in range(n)]),
                "sliding_window": _num(g("sliding_window_size")) or 2048}
    if mt == "gaggle":
        # GaggleConfig.window_sizes: S -> ceil(sequence_len / 4 / 128) * 128, L -> sequence_len, the last layer long
        # (configuration_gaggle.py:78-85); mask row - col <= window: window + 1 keys (modeling_gaggle.py:72-73)
        pat = str(g("window_pattern") or "SSSL").upper()
        if not pat or any(c not in "SL" for c in pat):
            return {}
        long_w = _num(g("sequence_len")) or 2048
        kinds = ["full" if pat[i % len(pat)] == "L" else "local" for i in range(n)]
        kinds[-1] = "full"
        return {"layer_types": sl(kinds), "sliding_window": -(-long_w // 4 // 128) * 128 + 1} if "local" in kinds else {}
    if mt == "jet_nemotron":
        # 'swa' layers (already local in _kind) see efficient_attention_config.swa.window_size keys
        # (modeling_jet_nemotron.py:348-353, mask j > i - W :207-210)
        swa = (g("efficient_attention_config") or {}).get("swa") if isinstance(g("efficient_attention_config"), dict) \
            else None
        w = _num(swa.get("window_size")) if isinstance(swa, dict) else 0
        return {"sliding_window": w} if w and "swa" in [str(x) for x in (g("layer_types") or [])] else {}
    if mt == "grug_moe":
        # grug_long_layer_flags: long iff (idx + 1) % global_every == 0 or idx == L - 1 (skyrl_train_models_grug_moe.py:
        # 50, :91-98); band key >= query - (W - 1) (:778)
        ge = _num(g("global_every")) or 4
        return {"layer_types": sl(["full" if ((i + 1) % ge == 0 or i == n - 1) else "local" for i in range(n)]),
                "sliding_window": _num(g("sliding_window")) or 2048}
    if mt == "plamo3":
        # is_full_attn = not (layer_idx + 1) % sliding_window_pattern (modeling_plamo.py:115-116), swa_mask
        # q - W < k (:426-431)
        p = _num(g("sliding_window_pattern")) or 8
        return {"layer_types": sl(["full" if (i + 1) % p == 0 else "local" for i in range(n)]),
                "sliding_window": _num(g("window_size")) or 2048}
    if mt == "edgerock":
        # fixed_memory_swa_layers become Qwen3_5FixedWindowAttention (modeling_edgerock.py:1036-1041), window
        # fixed_memory_swa_window (:827), band key_pos >= query_pos - window + 1 (:417-418)
        swa, lt = set(_int_list(g("fixed_memory_swa_layers"))), g("layer_types")
        if not swa or not isinstance(lt, list) or len(lt) != n:
            return {}
        return {"layer_types": [("sliding_attention" if (i in swa and x != "linear_attention") else x)
                                for i, x in enumerate(lt)],
                "sliding_window": _num(g("fixed_memory_swa_window")) or 1024}
    if mt == "ethoric":
        # attn_window on attn_window_layers only (Ethoric-10M/attention.py:180-185), q_pos - k_pos < window (:97-98)
        wl, w = set(_int_list(g("attn_window_layers"))), _num(g("attn_window"))
        return {"layer_types": sl(["local" if i in wl else "full" for i in range(n)]), "sliding_window": w} \
            if wl and w else {}
    if mt == "fabric" and str(g("architecture") or "fabric") != "full":
        # Bluebird-1.5: every layer's GQAAttention gets local_attention_window (fabric_runtime.py:269, :282, :310-314),
        # mask q - W < k <= q (:121-126). (attention_chunk_size is a query-chunking compute detail, :36, :177)
        return {"layer_types": ["sliding_attention"] * n,
                "sliding_window": _num(g("local_attention_window")) or 2048}
    return {}


def _lm_config(d):
    """The text-LM sub-config (_lm_unwrap) with its family-specific keys read into the standard ones (_normalize)."""
    if (_r3("breeze_backbone_config_unread") and isinstance(d, dict)
            and str(d.get("model_type") or "").lower() == "breeze"
            and str(d.get("backbone_model_type") or "breeze").lower() != "breeze"
            and _has_lm_dims(d.get("backbone_config"))):
        # round 3: Breeze-TTS builds its decoder from backbone_config (breeze_backbone_factory.py:105-135
        # AutoConfig.for_model(**backbone_config); :164-179 Qwen3DecoderLayer, Qwen3RMSNorm(backbone eps), Qwen3 RoPE);
        # the top-level Llama-style keys are unused. The lm_head is the wrapper's own Linear (breeze.py:923): the
        # wrapper's tie_word_embeddings (false) wins
        bb = d["backbone_config"]
        return _normalize({**bb, "model_type": str(bb.get("model_type") or d["backbone_model_type"]).lower(),
                           "tie_word_embeddings": d.get("tie_word_embeddings", False)})
    u = _lm_unwrap(d)
    if (_r3("no_model_type_longcat") and isinstance(u, dict) and isinstance(d, dict) and not u.get("model_type")
            and not d.get("model_type")
            and (set(d.get("architectures") or ()) | set(u.get("architectures") or ())) & LONGCAT_ARCHS):
        # round 3: the LongCat configs carry no model_type; their config classes say "longcat" (configuration_
        # longcat.py:95), transformers 5 has the family as longcat_flash (RENORM_FIXED, ROUTER_BIAS_TYPES, scmoe)
        u = {**u, "model_type": "longcat_flash"}
    return _normalize(u)


LONGCAT_ARCHS = frozenset({"LongcatFlashForCausalLM", "LongcatCausalLM", "LongcatForCausalLM"})


def _lm_unwrap(d):
    """Return the sub-config holding the text-LM transformer. Flat LM configs are
    returned as-is; multimodal / wrapper configs are unwrapped (recursively) to their
    nested text/llm config. model_type is carried down when the nested config omits it.

    A wrapper that MODIFIES a nested backbone (Looped-Qwen's qwen_config, MemT's backbone_configs)
    keeps only hidden_size / depth at the top and no heads: the backbone is unwrapped but the
    wrapper's model_type and loop fields win (the wrapper is the model, not its backbone)."""
    if isinstance(d, dict) and not (d.get("num_attention_heads") or d.get("n_head")):
        for k in _LM_WRAP_KEYS:
            sub = d.get(k)
            cand = _lm_unwrap(sub) if isinstance(sub, dict) else None
            if cand is not None and _has_lm_dims(cand):
                extra = {kk: d[kk] for kk in _LOOP_COUNT_KEYS if kk in d}
                return {**cand, **extra, "model_type": d.get("model_type") or cand.get("model_type")}
    if not isinstance(d, dict) or _has_lm_dims(d):
        return d
    for k in _LM_NEST_KEYS:
        sub = d.get(k)
        if isinstance(sub, dict):
            cand = _lm_unwrap(sub)
            if _has_lm_dims(cand):
                if not cand.get("model_type") and d.get("model_type"):
                    cand = {**cand, "model_type": d["model_type"]}
                return cand
    return d


# --- recurrent capacity (round 2, pass 4) ---------------------------------------
# A host's recurrent layers play a target's only if each is at least as large in every dimension the construction
# needs (literature review, recurrent / diffusion: no recurrent capacity was compared before, and >= 25 credited edges
# broke it -- Falcon-H1-7B SSM heads 24 < 32, Nemotron-H-56B head_dim 64 < 80, Nemotron-3-Puzzle d_state 96 < 128,
# Zamba2-7B d_state 64 < 128). dims_of's rc_* dims (0 = not present):
#   rc_h  heads (Mamba-1: the d_inner channels, each its own decay row; GDN: value heads)
#   rc_k  state per head (Mamba d_state N; GDN / KDA key head dim)
#   rc_v  value width per head (Mamba-2 head_dim P; GDN value head dim; Mamba-1: 1)
#   rc_kh key heads / B-C groups (GDN key heads; Mamba-2 n_groups)
#   rc_ng output-norm groups (Mamba-2 gated RMSNorm: n_groups, 1 = full width; 0 = no norm)
#   rc_conv causal conv width;  rc_rank low-rank dt / gate rank;  rc_x a family-specific extra (layers, meta tokens)
# and rc_rule, the construction the target's family admits:
#   RC_GE   every dim >= (zero-padded channels / states / ranks / heads are inert: zero in_proj rows give zero
#           post-conv inputs (silu(0) = 0), a zero B row leaves its state at 0, zero conv taps on the oldest positions
#           are a narrower causal conv). Mamba-1 (all variants), PLaMo-2, Falcon-H1 without the norm, LFM short conv.
#           Where an RMS norm reads the padded vector (Falcon-Mamba / Jamba / PLaMo-2 B, C, dt) its input is a linear
#           map's output: scaling those rows by sqrt(n_h / n_t) restores the mean square, the constant left on the
#           output folds into dt_proj / the D skip and out_proj -- exact with the same eps.
#   RC_M2   Mamba-2's gated RMSNorm over groups of w = I / G channels, eps = the model's single config eps (shared
#           with the stream norms; Zamba2: 1e-5 in code), and y = C h + D x has no free scale (x, B, C are silu outputs):
#           zero channels inside a group change its mean square, so each used host group holds m whole copies of one
#           target group's channels (the rms over m copies is the rms of one) -- w_h a multiple m of w_t, every host
#           head inside one target head (P_h divides m P_t; a copy may split a head or repeat channels), G_h >= G_t
#           (an all-zero host group outputs 0), N_h >= N_t (B / C zero-padded), conv >=. Divisibility composes.
#           (The eps lever -- the gated norm's eps is the config eps everywhere but Zamba2, so a zero-padded group
#           could set eps_h = eps_t w_t / w_h and rescale the residual stream for the stream norms sharing it -- is
#           left for pass 5 with the MLA latent one.)
#   RC_GDN  gated DeltaNet: q, k l2-normed and q scaled by dk^-1/2 in code (scale = 1 / sqrt(head_k_dim)), so a wider
#           key head rescales y by sqrt(dk_t / dk_h), which the per-head RMSNorm (config eps, shared) sees: dk EQUAL;
#           dv a multiple (replicated channels keep the per-head rms); key heads by count as KV heads
#           (kv_hosts_needed, replicated); conv >=.
#   RC_KDA  KDA: l2 q / k with the same 1 / sqrt(head_dim) scale -> head_dim equal; heads, conv, gate rank >= (a
#           full-rank gate plays f_b f_a; a zero-padded low rank is inert).
#   RC_EQ   every dim equal: the families whose dims enter fixed constants or that only one shape in the snapshot
#           has (MiniMax / Ling-2.5 lightning: the decay is fixed by (layer, L, H); RG-LRU block-diagonal gates;
#           RWKV; GLA; Jet; GDN-2; GKA; Hymba; Zamba-1 multi-head; SambaY).
# The universal host (rc_free = 1, elephantine/universal.py USSM) plays any of them at the join's dims: its RMS
# grouping is a weight (avg, block averages), its norm eps a buffer and its q / k scale a coefficient (q_coef), so
# zero-padded heads / states are inert there.
RC_GE, RC_M2, RC_GDN, RC_KDA, RC_EQ = 1, 2, 3, 4, 5
RC_DIMS = ("rc_h", "rc_k", "rc_v", "rc_kh", "rc_ng", "rc_conv", "rc_rank", "rc_x")


def _rec_dims(d):
    """dims_of's recurrent-capacity dims (RC_DIMS + rc_rule) of an LM config d (zeros for a softmax-only model)."""
    out = dict.fromkeys(RC_DIMS + ("rc_rule",), 0)
    cl = classify(d)
    if not cl or cl["mixer"] == "softmax":
        return out                                           # no recurrent layer: nothing to compare
    fam = _recurrent_family(d, cl["mixer"])
    mt = str(d.get("model_type") or "").lower()

    def g(*keys, default=0):
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        for k in keys:
            v = _class_default(mt, k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return default
    hidden = g("hidden_size", "d_model", "n_embd")
    la = d.get("linear_attn_config") if isinstance(d.get("linear_attn_config"), dict) else {}
    rank_auto = -(-hidden // 16) if hidden else 0          # time_step_rank / mamba_dt_rank "auto"
    if fam in ("mamba1", "mamba1_bcdt", "mamba1_bcdtw", "hymba", "mamba1_mh", "sambay"):
        if mt in ("mamba", "falcon_mamba"):                  # intermediate_size = the checkpoint's d_inner
            di = g("intermediate_size") or g("expand", default=2) * hidden
        else:
            di = g("mamba_expand", "expand", default=2) * hidden
        out.update(rc_h=di, rc_k=g("state_size", "mamba_d_state", "ssm_state_size", default=16), rc_v=1, rc_kh=1,
                   rc_conv=g("conv_kernel", "mamba_d_conv", "d_conv", default=4),
                   rc_rank=g("time_step_rank", "mamba_dt_rank", "dt_rank") or rank_auto,
                   rc_rule=RC_GE if fam in ("mamba1", "mamba1_bcdt", "mamba1_bcdtw") else RC_EQ)
        if fam == "mamba1_mh":
            out["rc_kh"] = g("n_mamba_heads", default=1)
        if fam == "hymba":
            out["rc_x"] = g("num_memory_tokens")
        if fam == "sambay":
            out["rc_x"] = g("num_hidden_layers")
        return out
    if fam in ("mamba2", "mamba2_nonorm", "mamba2_nbg"):
        nh = g("mamba_num_heads", "mamba_n_heads", "n_mamba_heads", "num_heads")
        P = g("mamba_head_dim", "mamba_d_head", "mamba_headdim", "head_dim" if mt == "mamba2" else "_")
        I = g("mamba_d_ssm") or g("mamba_expand", "expand", default=2) * hidden
        if not P and nh:
            P = I // nh                                      # mamba_d_head "auto"
        G = g("n_groups", "mamba_n_groups", "mamba_ngroups", "mamba_num_groups", default=1) or 1
        # the gated norm's groups: Nemotron-H / CENO / Zamba2 / Falcon-H1 group_size = I / n_groups; Bamba /
        # Granite-4.0-H / the HF Mamba2 eval path norm the full width (one group) whatever n_groups is
        ng = 1 if mt in ("bamba", "granitemoehybrid", "granite_switch", "mamba2") else G
        rule = RC_M2 if fam == "mamba2" else RC_GE
        if fam == "mamba2" and (ng != G or (nh % max(G, 1))):
            rule = RC_EQ                                     # a full-width norm over several B / C groups: no rule
        if fam == "mamba2_nonorm":
            ng, nh = 0, nh // max(G, 1)                      # GE on heads PER group (each host group serves one)
        out.update(rc_h=nh, rc_k=g("ssm_state_size", "mamba_d_state", "state_size", "mamba_state_dim", default=128),
                   rc_v=P, rc_kh=G, rc_ng=ng, rc_conv=g("conv_kernel", "mamba_d_conv", default=4), rc_rule=rule)
        return out
    if fam == "plamo2":
        out.update(rc_h=g("mamba_num_heads", default=64), rc_k=g("mamba_d_state", default=64),
                   rc_v=g("hidden_size_per_head", default=128), rc_kh=1, rc_conv=g("mamba_d_conv", default=4),
                   rc_rank=max(64, hidden // 16), rc_rule=RC_GE)  # PlamoMamba dt_dim = max(64, hidden // 16)
        return out
    if fam in ("gdn", "gdn_negeig", "gdn_sig"):             # (gdn_sig: round 3, gdn_out_gate_act -- the same dims)
        hv = g("linear_num_value_heads", default=0)
        hk = g("linear_num_key_heads", default=hv)
        out.update(rc_h=hv, rc_kh=hk or hv, rc_k=g("linear_key_head_dim"), rc_v=g("linear_value_head_dim"),
                   rc_conv=g("linear_conv_kernel_dim", default=4), rc_rule=RC_GDN)
        return out
    if fam in ("kda", "kda_lb", "kda_lb_negeig"):          # (kda_lb_negeig: round 3, Solar-Open2)
        if mt == "bailing_hybrid" or _bailing_v3_alias(d):   # (round 3: Edge0 / Ming-3 run Ling-3's code)
            H, hd, K = g("num_attention_heads"), g("head_dim"), g("short_conv_kernel_size", default=4)
            full = bool(d.get("no_kda_lora")) or d.get("use_kda_lora") is False
        elif la:
            H, hd = int(la.get("num_heads") or 0), int(la.get("head_dim") or 0)
            K = int(la.get("short_conv_kernel_size") or 4)
            full = bool(la.get("use_full_rank_gate"))
        else:                                                # GLM-5-Next top level, Youtu
            H = g("linear_num_heads", "linear_num_value_heads", "num_attention_heads")
            hd = g("linear_head_dim", "linear_key_head_dim", "head_dim")
            K = g("linear_conv_kernel_dim", "short_conv_kernel_size", default=4)
            full = bool(d.get("use_full_rank_gate", mt.startswith("youtu")))
        out.update(rc_h=H, rc_k=hd, rc_v=hd, rc_kh=H, rc_conv=K, rc_rank=hidden if full else hd, rc_rule=RC_KDA)
        return out
    if fam == "shortconv":
        out.update(rc_conv=g("conv_L_cache", default=3), rc_rule=RC_GE)
        return out
    if _r3("gla_rc_dims_unread") and fam in ("gla", "deltanet") and mt in ("gla", "delta_net"):
        # round 3: fla GLA / DeltaNet (gla.py:105-167, delta_net.py:116-156): key / value widths hidden * expand_k /
        # expand_v over num_heads (GLA's kv heads num_kv_heads or num_heads), short convs only under use_short_conv
        # (GLA's default off, DeltaNet's on), GLA's decay gate rank gate_low_rank_dim = 16 (the __init__ default:
        # modeling_gla.py never passes it); RC_EQ unchanged (equal dims only)
        gla = mt == "gla"
        fv = lambda k, dflt: d[k] if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool) else dflt
        ek, ev = fv("expand_k", 0.5 if gla else 1.0), fv("expand_v", 1.0)
        H = g("num_heads", default=4) or 4
        short = d.get("use_short_conv", not gla) is True
        out.update(rc_h=H, rc_k=round(hidden * ek) // H, rc_v=round(hidden * ev) // H,
                   rc_kh=(g("num_kv_heads") or H) if gla else H, rc_conv=(g("conv_size") or 4) if short else 0,
                   rc_rank=16 if gla else 0, rc_x=g("num_hidden_layers", "n_layer", "num_layers"), rc_rule=RC_EQ)
        return out
    # RC_EQ families: whatever shape the config states
    jet = ((d.get("efficient_attention_config") or {}).get("jet") or {}) if fam == "jet" else {}
    out.update(rc_h=int(jet.get("num_heads") or 0) or g("num_heads", "gdn_heads", "num_attention_heads"),
               rc_k=int(jet.get("head_dim") or 0) or g("gdn_head_dim", "lightning_head_dim", "head_dim", "lru_width"),
               rc_conv=int(jet.get("conv_size") or 0) or g("gdn_conv_size", "conv1d_width", "conv_size", "conv_L_cache"),
               rc_ng=g("group_norm_size"),
               rc_rank=g("decay_low_rank_dim", "gate_low_rank_dim", "a_low_rank_dim"),
               rc_x=g("num_hidden_layers", "n_layer", "num_layers"), rc_rule=RC_EQ)
    if _r3("jet_dims_unread") and fam == "jet" and isinstance(jet, dict):
        # round 3: the Jet block's value width head_dim * expand_v per head (jet_block.py:84-94, o_norm over head_v_dim
        # :146-148) and its dynamic-conv generator rank value_dim // dconv_generator_reduction (dynamic_conv.py:55-71;
        # JetBlockConfig defaults jet_block.py:44-49: expand_v 2.0, num_heads 6, head_dim 256, reduction 8)
        hd, H = int(jet.get("head_dim") or 256), int(jet.get("num_heads") or 6)
        ev = jet["expand_v"] if isinstance(jet.get("expand_v"), (int, float)) else 2.0
        red, vdim = int(jet.get("dconv_generator_reduction") or 0), int(H * hd * ev)
        out.update(rc_v=int(hd * ev), rc_rank=vdim // red if red else vdim)
    return out


def _rc_pair_ok(t, h):
    """Target recurrent dims t hosted by host dims h (dicts of RC_DIMS + rc_rule), the target's rule."""
    rule = t["rc_rule"]
    ge = lambda *ks: all(h[k] >= t[k] for k in ks)
    if rule == RC_GE:
        return ge(*RC_DIMS)
    if rule == RC_EQ:
        return all(h[k] == t[k] for k in RC_DIMS) and h["rc_rule"] == RC_EQ
    if rule == RC_M2:
        if h["rc_rule"] != RC_M2 or not t["rc_kh"] or not h["rc_kh"]:
            return h["rc_rule"] == RC_M2 and all(h[k] == t[k] for k in RC_DIMS)
        wt, wh = t["rc_h"] * t["rc_v"] // t["rc_kh"], h["rc_h"] * h["rc_v"] // h["rc_kh"]
        if not wt or wh % wt:
            return False
        m = wh // wt
        return (h["rc_kh"] >= t["rc_kh"] and (m * t["rc_v"]) % max(h["rc_v"], 1) == 0
                and ge("rc_k", "rc_conv"))
    if rule == RC_GDN:
        if h["rc_rule"] != RC_GDN or h["rc_k"] != t["rc_k"] or not t["rc_v"] or h["rc_v"] % t["rc_v"]:
            return False
        kt, kh = max(t["rc_kh"], 1), max(h["rc_kh"], 1)
        need = kv_hosts_needed(kt, max(t["rc_h"] // kt, 1), max(h["rc_h"] // kh, 1), 1)
        return kh >= need and ge("rc_conv")
    if rule == RC_KDA:
        return h["rc_rule"] == RC_KDA and h["rc_k"] == t["rc_k"] and ge("rc_h", "rc_conv", "rc_rank")
    return False


def _rc_pair_ok_lever(t, h):
    """The recurrent families under the eps lever (round 2, pass 5): the output norm's eps is config, so a zero-
    padded wider group / head is reproduced by eps_h = eps_t r (r = the padded mean-square ratio; the stream
    rescaled for the stream norms sharing the eps). Each rule is implied by its strict one and composes.
      Mamba-2: a host group holds ONE target group zero-padded, each target head on ceil(P_t / P_h) host heads of its
        group (one dt / A): host heads per group >= target heads per group x ceil(P_t / P_h), G_h >= G_t, N, conv >=.
      GDN: the q scale 1 / sqrt(dk) and the zero-padded value channels are constant factors on the per-head norm's
        input: dk_h >= dk_t, dv_h >= dv_t (were equal / a multiple), key heads by count, conv >=.
      KDA: the same for its one head dim: hd_h >= hd_t, heads, conv, gate rank >=."""
    rule = t["rc_rule"]
    if rule != h["rc_rule"]:
        return False
    if rule == RC_M2:
        if not t["rc_kh"] or not h["rc_kh"]:
            return False
        hpg_t, hpg_h = t["rc_h"] // t["rc_kh"], h["rc_h"] // h["rc_kh"]
        split = -(-t["rc_v"] // max(h["rc_v"], 1))
        return (h["rc_kh"] >= t["rc_kh"] and hpg_h >= hpg_t * split
                and h["rc_k"] >= t["rc_k"] and h["rc_conv"] >= t["rc_conv"])
    if rule == RC_GDN:
        if h["rc_k"] < t["rc_k"] or h["rc_v"] < t["rc_v"]:
            return False
        kt, kh = max(t["rc_kh"], 1), max(h["rc_kh"], 1)
        need = kv_hosts_needed(kt, max(t["rc_h"] // kt, 1), max(h["rc_h"] // kh, 1), 1)
        return kh >= need and h["rc_conv"] >= t["rc_conv"]
    if rule == RC_KDA:
        return all(h[k] >= t[k] for k in ("rc_k", "rc_v", "rc_h", "rc_conv", "rc_rank"))
    return False


def _r3_rec_nest(rf_t, rf_h):
    """Round 3 (mamba1_lever): the lever nest (R3_FEAT_NEST) a host's recurrent family set rf_h uses for a one-family
    target rf_t it does not contain, else None."""
    if len(rf_t) != 1:
        return None
    (x,) = tuple(rf_t)
    nest = _r3_nest(x)
    return x if nest and any(p in rf_h for p, _ in nest) else None


def _rc_nonorm_on_gated(t, h):
    """fr-mixer m2_nonorm_lever: a norm-free Mamba-2 target t (rc_h = heads PER group, _rec_dims) on a gated-norm
    Mamba-2 host h (rc_h = all its heads) whose norm the residual-scaled eps lever makes linear. No norm groups are left
    to respect, so zero padding is inert (mamba_channel_inert, mamba_state_pad): a target group (its B / C) takes whole
    host groups (B / C copied), each target head ceil(P_t / P_h) host heads of one group (one dt / A / D), channels
    zero-padded -- G_h >= G_t ceil(hpg_t ceil(P_t / P_h) / hpg_h); d_state, conv >=."""
    G_t, G_h = max(t["rc_kh"], 1), max(h["rc_kh"], 1)
    hpg_h = h["rc_h"] // G_h
    if not hpg_h or not h["rc_v"]:
        return False
    split = -(-t["rc_v"] // h["rc_v"])
    need = -(-(t["rc_h"] * split) // hpg_h)
    return G_h >= G_t * need and h["rc_k"] >= t["rc_k"] and h["rc_conv"] >= t["rc_conv"]


def _rec_ok(P, lever=False):
    """[i, j]: host P[j]'s recurrent layers are large enough for target P[i]'s (_rc_pair_ok), where the target has
    recurrent layers and the host carries their family (rf_ features; feats_ok walls the rest); the universal host
    (dims rc_free) at its join dims. Also Zamba2 -> Zamba2: the host's shared blocks a multiple of the target's
    (block i % num_mem_blocks: each host block stands for one target block), adapter rank and attention adapters >=."""
    import numpy as np
    n = len(P)
    ok = np.ones((n, n), bool)
    rf = [frozenset(x for x in p["slots"].get("feats", "").lstrip("*").split("+") if x.startswith("rf_")) for p in P]
    uni = [p["slots"].get("feats", "").startswith("*") for p in P]
    rc = [{k: int(p["dims"].get(k, 0)) for k in RC_DIMS + ("rc_rule",)} for p in P]
    free = [bool(p["dims"].get("rc_free")) for p in P]
    zb = [(int(p["dims"].get("zamba_mem", 0)), int(p["dims"].get("zamba_rank", 0)),
           int(p["dims"].get("zamba_attn_adapt", 0))) for p in P]
    tgt = [i for i in range(n) if rf[i] and rc[i]["rc_rule"]]
    # round 2, pass 5: the eps lever on a lever-capable Mamba-2 host whose gated-norm eps is the config eps
    lev = [bool(lever and p["dims"].get("eps_lever") and p["dims"].get("rec_eps_cfg")) for p in P]
    for i in tgt:
        for j in range(n):
            if i == j or free[j]:
                continue
            if not (uni[j] or rf[i] <= rf[j]):
                nest = _r3_rec_nest(rf[i], rf[j])     # round 3 / fr-mixer: a lever nest (feats_ok passes it, payable)
                if nest == "rf_mamba2_nonorm":         # m2_nonorm_lever: the residual-scaled config-eps lever
                    ok[i, j] = lev[j] and _rc_nonorm_on_gated(rc[i], rc[j])
                elif nest == "rf_mamba1":              # mamba1_lever: payable, the eps lever
                    ok[i, j] = bool(lever and P[j]["dims"].get("eps_lever")) and _rc_pair_ok(rc[i], rc[j])
                continue                                   # a family wall (feats_ok), or mode="mixgap"'s near-gap
            ok[i, j] = _rc_pair_ok(rc[i], rc[j]) or (lev[j] and _rc_pair_ok_lever(rc[i], rc[j]))
    for i in range(n):
        if not zb[i][0]:
            continue
        for j in range(n):
            if i != j and zb[j][0]:
                ok[i, j] &= zb[j][0] % zb[i][0] == 0 and zb[j][1] >= zb[i][1] and zb[j][2] >= zb[i][2]
    return ok


def _zamba_dims(d):
    """Zamba / Zamba2's shared transformer blocks (num_mem_blocks), the per-invocation LoRA rank (gate_up always in
    Zamba2; q / k / v under use_shared_attention_adapter); zeros elsewhere."""
    mt = str(d.get("model_type") or "").lower()
    if mt == "zamba":
        return {"zamba_mem": 1, "zamba_rank": 0, "zamba_attn_adapt": 0}
    if mt == "zamba2":
        rank = int(_num(d.get("adapter_rank", _class_default(mt, "adapter_rank") or 128)))
        return {"zamba_mem": int(_num(d.get("num_mem_blocks")) or 1), "zamba_rank": rank,
                "zamba_attn_adapt": rank if d.get("use_shared_attention_adapter") else 0}
    return {"zamba_mem": 0, "zamba_rank": 0, "zamba_attn_adapt": 0}


_EPS_KEYS = ("rms_norm_eps", "layer_norm_eps", "norm_eps", "layer_norm_epsilon", "norm_epsilon", "rms_norm_epsilon",
             "layernorm_epsilon")


def _q_bias(d):
    """Does the attention's query projection have a bias? Explicit keys first, then the model type (graph/bos_rules).
    Round 3 (zoo_phi3small_biases): a type whose code fixes the bias whatever its keys say (BIAS_BY_CODE_TYPES) first."""
    if str(d.get("model_type") or "").lower() in BIAS_BY_CODE_TYPES:
        return True
    for k in _QBIAS_KEYS:
        if k in d and isinstance(d[k], bool):
            return d[k]
    if isinstance(d.get("no_bias"), bool) and d.get("model_type") not in ("qwen",):
        return not d["no_bias"]
    return d.get("model_type") in QBIAS_TYPES


def _ffn_bias(d):
    """Does the FFN input (gate / up) projection have a bias (graph/act_width.host_gate_bias)?"""
    mt = str(d.get("model_type") or "").lower()
    if mt in BIAS_BY_CODE_TYPES:
        return True                                        # (round 3, zoo_phi3small_biases)
    if "no_bias" in d:
        return not bool(d["no_bias"])
    for f in ("mlp_bias", "use_bias", "add_bias_linear", "enable_bias", "bias"):
        if f in d and isinstance(d[f], bool):
            return d[f]
    return mt in _GATE_BIAS_TYPES


_GATE_BIAS_TYPES = frozenset({"gpt2", "gpt_neox", "gpt_neox_japanese", "gptj", "gpt_neo", "gpt_bigcode", "bloom",
                              "codegen", "opt", "phi", "persimmon", "xglm", "biogpt", "starcoder2", "gpt_oss", "jais"})
# round 3: GPT-NeoX-Japanese's dense_h_to_4h is bias=False (modeling_gpt_neox_japanese.py:312; bias_qkv_table_wrong);
# RecurrentGemma's gate / up projections are bias=True (modeling_recurrent_gemma.py:489-490; bias_out_by_code_unlisted)
_GATE_BIAS_TYPES = _R3Set(_GATE_BIAS_TYPES, add={"bias_out_by_code_unlisted": {"recurrent_gemma"}},
                          drop={"bias_qkv_table_wrong": {"gpt_neox_japanese"}})
# round 3, the zoo's params triage (zoo_phi3small_biases): Phi-3-small's every Linear keeps its default bias whatever
# the config says (its configs carry attention_bias false, which the code never reads): query_key_value / dense
# (modeling_phi3_small.py:208-209), Phi3SmallMLP up_proj h -> 2 I / down_proj (:163-164); the checkpoints hold every
# bias. p5 already gave it bias_qkv / bias_out (BIAS_QKV_TYPES / BIAS_OUT_TYPES); the q bias and the FFN input bias
# (the hosts' dims q_bias / ffn_bias) read the dead key. Checked before the keys in _q_bias / _ffn_bias.
BIAS_BY_CODE_TYPES = _R3Set((), add={"zoo_phi3small_biases": {"phi3small"}})


def _rope_kind(d):
    """The RoPE frequency type(s) beyond the default schedule ("" = every table the default theta^(-2i/d), what a
    fixed-rotary host reproduces with rope_theta alone, Lean defaultFreq_match): the rope_type / type names, sorted;
    on a fixed-rotary family suffixed with its model type (its own code computes them, so only that family's code
    is known to)."""
    kinds = set()
    for key in ("rope_parameters", "rope_scaling"):
        rp = d.get(key)
        if isinstance(rp, dict):
            per = _rope_layer_types(rp)
            for v in (per.values() if per else [rp]):
                t = (v or {}).get("rope_type") or (v or {}).get("type")
                if t not in (None, "default"):
                    kinds.add(str(t))
    if not kinds:
        return ""
    mt = str(d.get("model_type") or "").lower()
    return "+".join(sorted(kinds)) + ("" if mt in ROPE_PAIRS_FREE_TYPES else "@" + mt)


def _kv_shapes(d, nh, nkv, hd):
    """Every per-layer attention shape (the top-level one and _attn_variants'), as sorted (kv heads, q per kv, head
    dim) tuples, when there are several; () for MLA (every head its own K / V from the latent) and single-shape
    stacks (kv_heads_fit then reads kv_heads / q_per_kv / head_dim)."""
    tab = _kv_shape_table(d, nh, nkv, hd)
    return tuple(sorted(tab)) if len(tab) > 1 else ()


# round 3 (v_head_dim_narrow): the families whose V heads are narrower / wider than their q / k heads, read only there:
# MiMo-V2 / -V2-Flash / Naive-N0.5-Flash v_head_dim / swa_v_head_dim (modeling_mimo_v2_flash.py:307-326, :353: V 128
# wide under 192-wide q / k), Hymba v_head_dim (modeling_hymba.py:503, :524-530), Needle qk_head_dim / value_head_dim
# (cactus architecture.py:174-175; _normalize sets head_dim = qk_head_dim)
V_WIDTH_TYPES = _R3Set((), add={"v_head_dim_narrow": {"mimo_v2", "mimo_v2_flash", "naive_n05_flash", "needle",
                                                      "hymba"}})


def _v_head(d, hd):
    """The V width per KV head of the top-level shape (= the q / k head_dim except in V_WIDTH_TYPES; MLA: head_dim)."""
    mt = str(d.get("model_type") or "").lower()
    if mt not in V_WIDTH_TYPES or d.get("kv_lora_rank"):
        return int(hd)
    return int(_num(d.get("value_head_dim")) or _num(d.get("v_head_dim")) or hd)


def _per_layer_heads(d, nh, nkv, hd):
    """Round 3 (per_layer_head_shapes): per-layer (query heads, KV heads, head dim) from a per-layer list the code reads
    -- Laguna num_attention_heads_per_layer (modeling_laguna.py:427; 8 KV heads everywhere), DeciLM-7B
    num_key_value_heads_per_layer (modeling_decilm.py:35-38), Domyn-Edge head_sizes (configuration_domynedge.py:103-110,
    modeling_domynedge.py:203-221: q / k / v / o at heads x the layer's head size) -- else None."""
    if not _r3("per_layer_head_shapes"):
        return None
    qpl, kpl, hs = d.get("num_attention_heads_per_layer"), d.get("num_key_value_heads_per_layer"), d.get("head_sizes")
    if isinstance(qpl, list) and qpl:
        return [(_num(q) or nh, nkv, hd) for q in qpl]
    if isinstance(kpl, list) and kpl:
        return [(nh, _num(k) or nkv, hd) for k in kpl]
    if isinstance(hs, list) and hs:
        return [(nh, nkv, _num(x) or hd) for x in hs]
    return None


def _layer_kind_list(d):
    """Per-layer 'full' / 'local' / ... from layer_types (else MiMo's hybrid_layer_pattern, 0 = full), or None."""
    lt = d.get("layer_types")
    if isinstance(lt, list) and lt and all(isinstance(x, str) for x in lt):
        return [_kind(x) for x in lt]
    hlp = d.get("hybrid_layer_pattern")
    if isinstance(hlp, list) and hlp and all(x in (0, 1) for x in hlp):
        return ["full" if x == 0 else "local" for x in hlp]
    return None


def _kv_shape_table(d, nh, nkv, hd):
    """{(kv heads, q per kv, q / k head dim, v head dim): config layers of that shape} over the attention layers ({} for
    MLA: every head its own K / V from the latent). The top-level shape plus _attn_variants' (round 2, pass 5) and,
    round 3, the per-layer lists (per_layer_head_shapes) and the V width (v_head_dim_narrow; = the head dim elsewhere).
    The layer counts (per_layer_head_shapes: a host offers each shape only on its own layers) follow the code's layer
    assignment: a per-layer list layer by layer; Step-3.5's attention_other_setting and the swa_* shapes on the sliding
    layers; Gemma-4's global_head_dim on the full layers; per_layer_config on its listed layer indices. A shape whose
    layers cannot be placed counts 0 (dims_of then spreads the attention layers evenly: _kv_shape_counts)."""
    if d.get("kv_lora_rank") or d.get("swa_kv_lora_rank") or not nkv or not hd:
        return {}
    vw = _r3("v_head_dim_narrow")
    mt = str(d.get("model_type") or "").lower()
    qpk = lambda q, k: max(1, int(q) // max(int(k), 1))
    v0 = _v_head(d, hd) if vw else int(hd)
    pl = _per_layer_heads(d, nh, nkv, hd)
    if pl:
        tab = {}
        for q, k, h in pl:
            s = (int(k), qpk(q, k), int(h), int(h) if h != hd else v0)
            tab[s] = tab.get(s, 0) + 1
        return tab
    base = (int(nkv), qpk(nh, nkv), int(hd), v0)
    kinds = _layer_kind_list(d)
    n = len(kinds) if kinds else (_num(d.get("num_hidden_layers")) or 0)
    lay = [base if (kinds is None or kinds[i] in ("full", "local")) else None for i in range(n)]
    unplaced = []
    variants = _attn_variants(d, tagged=True)
    aos = d.get("attention_other_setting")
    ghd = _num(d.get("global_head_dim"))
    plc = d.get("per_layer_config") if isinstance(d.get("per_layer_config"), dict) else {}
    order = ((["aos"] if isinstance(aos, dict) and aos.get("head_dim") else []) + (["global"] if ghd else [])
             + [("plc", k) for k, v in plc.items() if isinstance(v, dict) and v.get("head_dim")]
             + (["swa"] if (not d.get("swa_kv_lora_rank") and d.get("swa_head_dim")) else [])
             + (["ernie"] if _ernie_mla(d) else []))
    for (vh, vhd, vkv, _), src in zip(variants, order):
        if not (vhd and vkv and vkv % vhd == 0):
            continue
        k = int(vkv // vhd)
        v = int(vhd)
        if vw and src == "swa" and mt in V_WIDTH_TYPES:
            v = int(_num(d.get("swa_v_head_dim")) or _num(d.get("v_head_dim")) or vhd)
        s = (k, qpk(vh, k), int(vhd), v)
        if src in ("aos", "swa", "global") and kinds:
            want = "full" if src == "global" else "local"
            for i in range(n):
                if kinds[i] == want:
                    lay[i] = s
        elif isinstance(src, tuple) and str(src[1]).isdigit() and int(src[1]) < n:
            lay[int(src[1])] = s
        else:
            unplaced.append(s)
    tab = {base: 0}                                        # (the top-level shape is always listed, as in pass 5)
    for s in lay:
        if s is not None:
            tab[s] = tab.get(s, 0) + 1
    for s in unplaced:
        tab.setdefault(s, 0)
    return tab


def _kv_shape_counts(tab, shapes, total):
    """Round 3 (per_layer_head_shapes): the attention layers of each shape in `shapes` (sorted kv_shapes), scaled to
    `total` (the point's FU + FT + LU + LT eff count: the layers kind_match counts) by the largest remainder; a table
    whose layers could not be placed spreads them evenly. () when the counted fit is off."""
    if not _r3("per_layer_head_shapes") or not shapes:
        return ()
    c = [tab.get(s, 0) for s in shapes]
    if sum(c) <= 0:
        c = [1] * len(shapes)
    tot = sum(c)
    raw = [x * total / tot for x in c]
    out = [int(r) for r in raw]
    for i in sorted(range(len(c)), key=lambda i: -(raw[i] - out[i]))[:total - sum(out)]:
        out[i] += 1
    return tuple(out)


def _pass5_dims(d, outer=None):
    """The round-2 pass-5 dims of an LM config d (0 where they cannot matter, so the dedup key splits no point
    needlessly): the host's config levers, the start-token / rescue facts."""
    out = dict(win_free=0, rot_frac_free=0, rope_kind="", softcap_free=0, qk_switch=0, eps_lever=0, q_bias=0,
               ffn_bias=0, lm_head_bias=0, ln_beta=0, loop_plain=0, loop_norm=0, loop_vbias=0, rms_ok=0, pos_table=0,
               sc_norm_off=0,
               rec_eps_cfg=0)
    cl = classify(d)
    if not cl:
        return out
    outer = outer or {}
    mt = str(d.get("model_type") or "").lower()
    feats = set(filter(None, cl.get("feats", "").split("+")))
    mla = bool(d.get("kv_lora_rank"))
    if mt in WINDOW_FREE_TYPES and (cl["window"] == "local"):
        out["win_free"] = 1
    if cl["pos"] in ("rope", "partial_rope"):
        # (round 3, rope_cap_zero_tail_table: the remote LFM3 rotary reads partial_rotary_factor over a whole-head table
        # with a zero tail -- the factor only moves where the tail starts, Lean partial_rotary_raise_buffer)
        out["rot_frac_free"] = int((mt in ROT_FRAC_FREE_TYPES or (_r3("rope_cap_zero_tail_table")
                                                                  and _zero_tail_rope(d, outer))) and not mla)
        if mt not in ROPE_PAIRS_FREE_TYPES:                 # (read only between fixed-rotary points: fixed_rot_fits)
            out["rope_kind"] = _rope_kind(d)
    if mt in SOFTCAP_FREE_TYPES and cl["softcap"] != "no":
        out["softcap_free"] = 1
    if mt in QKNORM_SWITCH_TYPES and cl["qknorm"] != "no":
        out["qk_switch"] = 1
    eps = next((d[k] for k in _EPS_KEYS if d.get(k) is not None), None)
    # (the fp16 condition is a RUN condition, not an architecture's: an emulator built by weight-setting runs in the
    # fp32 / bf16 the lever was measured in whatever dtype a checkpoint was stored in. Read as the stored dtype it
    # walled fp16-stored Qwen3 / Gemma-4 finetunes from their own architecture -- 885 + 11 transitivity violations
    # through bf16 siblings -- so it is recorded as the lever's run condition only: run the emulator in bf16 / fp32.
    # Round 3, a design decision: this run-dtype reading stays; torch_dtype is not read here.)
    switchable = (cl["qknorm"] in EPS_LEVER_QKNORM or feats & EPS_LEVER_FEATS or mla
                  or _rec_dims(d).get("rc_rule") in (RC_M2, RC_GDN, RC_KDA))
    # round 3 (mamba1_lever): the dt / B / C norms of Falcon-Mamba (eps mixer_rms_eps, modeling_falcon_mamba.py) and
    # Jamba (eps rms_norm_eps) switch off by the lever where their eps > 0
    m1n = bool(feats & {"rf_mamba1_bcdt", "rf_mamba1_bcdtw"}) and _r3("mamba1_lever")
    if m1n and mt == "falcon_mamba":
        me = d.get("mixer_rms_eps", 1e-6)
        m1n = not (isinstance(me, (int, float)) and me <= 0)
    switchable = switchable or m1n
    if switchable and not (isinstance(eps, (int, float)) and eps <= 0):
        out["eps_lever"] = 1
    out["q_bias"] = int(bool(_q_bias(d)))
    out["ffn_bias"] = int(bool(_ffn_bias(d)))
    if mt in SC_NORM_OFF_TYPES and "selfcond_norm" in feats:
        out["sc_norm_off"] = 1                             # fr-resnorm (fr_selfcond_norm_host)
    if "LayerNorm" not in branches(cl["norm"]) and (d.get("lm_head_bias") is True or mt in LM_HEAD_BIAS_TYPES
                                                    or (mt == "ernie4_5_moe" and d.get("use_bias") is True)):
        out["lm_head_bias"] = 1
    # a LayerNorm beta: the bias_out feature's "affine LayerNorm beta" (MPT no_bias: LPLayerNorm without one)
    if (cl["norm"] == "LayerNorm" and "bias_out" in feats and mt not in LN_NO_BETA_TYPES and d.get("no_bias") is not True
            and d.get("use_norm_bias", True) is not False and d.get("norm_bias", True) is not False):
        out["ln_beta"] = 1
    if cl["loop"] == "looped":
        out["loop_plain"] = int(mt in LOOP_PLAIN_TYPES)
        out["loop_norm"] = int(mt in LOOP_NORM_BETWEEN_TYPES)
        out["loop_vbias"] = out["q_bias"]                  # q / k / v biases come together (Qwen2-style)
    if (out["ln_beta"] or cl["block"] == "parallel") and PRENORM_RMS_RANGE.get(mt, 1e9) <= PRENORM_D_MAX:
        out["rms_ok"] = 1
    if cl["pos"] == "learned_abs" or (cl["pos"] == "sinusoidal" and _r3("fp_sinusoid_table")):
        # (dims_of sets the table's length; fr-positions fp_sinusoid_table: a sinusoid table is a fixed table of the
        # rows its context reads, its length the stated context as a learned one's)
        out["pos_table"] = 1
    if _rec_dims(d).get("rc_rule") in (RC_M2, RC_GDN, RC_KDA) and mt in REC_EPS_CFG_TYPES:
        out["rec_eps_cfg"] = 1
    return out


# round 3 (embedding_norm): a norm applied to the token embedding before layer 0, fixed in code (no key): BLOOM
# word_embeddings_layernorm (affine LN, modeling_bloom.py:425 / :499), RWKV-4 / 5 / 6 layer-0 pre_ln (modeling_rwkv.py:332
# / :342, modeling_rwkv5.py:406), fla RWKV-7 pre_norm iff norm_first; weight-free RMS: nanochat_gpt / nanochat (an
# "additional norm before the layers", modeling_nanochat.py:406), gaggle, sn38-nanochrono, Muse-Glimmer
# (MuseGlimmerTextNormedEmbedding, with_scale=False), NeoMME; weighted RMS: Inkling; Grug-MoE embed_norm + embed_gated_norm
# (skyrl_train_models_grug_moe.py:892-893, :974). (LFM2's embedding_norm is its FINAL norm: not this.)
EMB_NORM_BY_CODE = _R3Dict({}, add={"embedding_norm": {
    **{t: "ln" for t in ("bloom", "rwkv", "rwkv5", "rwkv6")},
    **{t: "rms_free" for t in ("nanochat_gpt", "nanochat", "gaggle", "sn38-nanochrono", "muse_glimmer",
                               "muse_glimmer_text", "neomme")},
    "inkling": "rms", "grug_moe": "rms_gated"}})
# the embedding-norm kinds a host's (always-on) norm can be set to: weight 1 is the weight-free RMS norm (Lean
# qkn_gain_one: gain 1), a zero fc2 makes Grug's gate 1/2, folded into the weight (Lean out_gate_off / shared_gate_half)
EMB_NEST = {"": ("",), "ln": ("ln",), "rms_free": ("rms_free",), "rms": ("rms", "rms_free"),
            "rms_gated": ("rms_gated", "rms", "rms_free")}
# The untied fold E'[v] = N(E[v]) (a target's embedding norm folded into the host's input table, exact for any N): needs
# theorem emb_fold_untied (PROOFS.md has none), so a target's embedding norm is a WALL on every host whose own kind does
# not nest it until that lands (then: True, and an untied target -- or a tied one on an untied host -- folds).
EMB_FOLD_UNTIED = False


R3_DIMS = ("emb_rank", "emb_norm", "norm_groups", "gnorm_rank", "gnorm_sites")


def _r3_dims(d):
    """Round 3, batch 1 dims of an LM config d (neutral values elsewhere, so the dedup key splits nothing needlessly):
      emb_rank     OPT's word_embed_proj_dim where it differs from hidden (factorized_embedding_proj; 0 = none)
      emb_norm     the embedding norm's kind, EMB_NORM_BY_CODE ("" = none; embedding_norm)
      norm_groups  K2-Horizon / Jais-ASR's grouped RMSNorm groups, layernorm_num_groups (1 = full width; grouped_rmsnorm)
      gnorm_rank / gnorm_sites  the gated pre-norm's rank and sites (gated_prenorm_lowrank; 0 / "" = none)"""
    mt = str(d.get("model_type") or "").lower()
    if mt == "rwkv7":
        en = "ln" if (_r3("embedding_norm") and d.get("norm_first", True) is not False) else ""
    else:
        en = EMB_NORM_BY_CODE.get(mt, "")
    ng = 1
    if _r3("grouped_rmsnorm") and mt in ("k2_horizon", "jais_asr"):
        ng = max(1, _num(d.get("layernorm_num_groups")) or 1)
    gr, gs = _gnorm_dims(d)
    return {"emb_rank": _emb_rank(d) if _r3("factorized_embedding_proj") else 0, "emb_norm": en,
            "norm_groups": ng, "gnorm_rank": gr, "gnorm_sites": gs}


def _emb_rank(d):
    """OPT's factorized embedding (modeling_opt.py:289-300, :368-369, :393, :454): word_embed_proj_dim != hidden builds
    project_in / project_out, so the width-h input table is E P_in^T and the head E P_out (rank <= wpd); 0 elsewhere."""
    if str(d.get("model_type") or "").lower() != "opt":
        return 0
    wpd, h = _num(d.get("word_embed_proj_dim")), _num(d.get("hidden_size"))
    return wpd if wpd and h and wpd != h else 0


def _ffn_width_by_code(d, mt, hidden):
    """Round 3, batch 3b (ffn_width_keys_unread): a dense FFN width the code computes when no width key is set (0 =
    none): RWKV-5 int(hidden * 3.5 // 32 * 32) (configuration_rwkv5.py:110 forces intermediate_size None,
    modeling_rwkv5.py:363-366), Ivme int(hidden * ffn_mult * 2 / 3) up to a multiple of 8 (modeling_ivme.py:93-94),
    fla GatedMLP int(hidden * hidden_ratio * 2 / 3) up to a multiple of 256 (fla/mlp.py:46-50: gla, delta_net, hgrn_bit)."""
    if not hidden:
        return 0
    if mt == "rwkv5":
        return int((hidden * 3.5) // 32 * 32)
    if mt == "ivme" and isinstance(d.get("ffn_mult"), (int, float)):
        w = int(hidden * d["ffn_mult"] * 2 / 3)
        return (w + 7) // 8 * 8
    if mt in ("gla", "delta_net", "hgrn_bit"):
        hr = d.get("hidden_ratio") if isinstance(d.get("hidden_ratio"), (int, float)) else 4
        w = int(hidden * (hr or 4) * 2 / 3)
        return 256 * ((w + 255) // 256)
    return 0


def _loop_layers(d):
    """(n_layers, param_layers, eff_layers) of an LM config d: the config's layer count, the weight-DISTINCT layers and
    the unrolled forward depth (equal for a plain stack; a looped / recurrent-depth model has few unique layers -- its
    block -- but a deep unrolled forward pass, block x recurrence)."""
    g = lambda *keys: next((_num(d[k]) for k in keys if d.get(k) not in (None, [], "")), 0)
    n_layers = g("num_hidden_layers", "n_layer", "num_layers",
                 "num_transformer_layers", "n_layers") or _layer_count(d)
    if _looped(d):
        rec = g("mean_recurrence", "num_recurrence", "recurrence") or 1
        blk = g("n_layers_in_recurrent_block")
        if d.get("model_type") == "albert":
            param_layers = max(1, g("num_hidden_groups")) * max(1, g("inner_group_num"))
            eff_layers = n_layers or param_layers
        elif blk:
            param_layers, eff_layers = blk, blk * rec
        elif not g("mean_recurrence", "num_recurrence", "recurrence"):
            param_layers, eff_layers = _loop_depth(d, n_layers)
            if not n_layers and _r3("loop_depth_unread") and str(d.get("model_type") or "").lower() == "cfrd":
                n_layers = param_layers                     # round 3: CFRD has no layer-count key (its cells)
        else:
            param_layers, eff_layers = n_layers, n_layers * rec
    else:
        param_layers = eff_layers = n_layers
    return n_layers, param_layers, eff_layers


def _loop_dims(d):
    """(param_layers, eff_layers) of an LM config d (_loop_layers)."""
    return _loop_layers(d)[1:]


# --- round 3 (max_context_unknown): the max context, read, never defaulted ----------------------------------------
# dims max_context (and eff_window, the full-context reach of a layer with no window) is a READ value (int > 0), None =
# UNKNOWN (no source: not a guess), or CTX_UNBOUNDED (a model whose code has no limit). Before, a config with no position
# key got 4096 (44 snapshot repos: Kimi-Linear's model_max_length 1048576, LLaMA-1's max_sequence_length 2048, Falcon's
# class default 2048, Mamba with no limit at all). Readers handle the three cases explicitly: emul_matrix's context axis
# (_ctx_codes / _ctx_ge), the Z3 model (verify/z3_relation.py ctx_ge), the start-token rule (a learned-abs table of
# unknown length pays nothing: pos_table 0 and the rule wants pos_table > 0), minimal_emulator.join.
CTX_UNBOUNDED = 1 << 40           # above every read value (those are clamped to 1 << 20, the placeholder clamp)
CTX_UNKNOWN = -1                  # the code emul_matrix / Z3 compare an unknown context as (its own value)
CTX_POS_KEYS = ("max_position_embeddings", "max_seq_len", "n_positions", "seq_length")    # the usual position keys
# rope types whose original_max_position_embeddings x factor is the extended window (YaRN / DeepSeek-YaRN, dynamic NTK,
# linear interpolation, LongRoPE); never llama3, whose factor shapes the frequencies (Llama-3.2's 8192 x 32 is not its
# 131072 limit). The order and the lists follow graph/arch_metrics._max_context (the zoo side, branch zoo-errors)
CTX_ROPE_EXTEND = ("yarn", "deepseek_yarn", "dynamic", "linear", "longrope")
# other spellings of the trained / supported length, LM config then the top level: model_max_length (Kimi-Linear 1048576:
# KimiLinearConfig's attribute_map sends max_position_embeddings there, configuration_kimi_linear.py:68; Baichuan's
# alibi mask), OLMo-1 / LLaMA-1 max_sequence_length, nanochat / gaggle sequence_len, RWKV context_length (the CUDA
# kernel's Tmax: modeling_rwkv.py:46-85 refuses longer inputs), max_seq_length, GPT-2 n_ctx
CTX_ALT_KEYS = ("model_max_length", "max_sequence_length", "sequence_len", "context_length", "max_seq_length", "n_ctx")
# family-specific length keys a generic reading would mis-take (block_size is LLaDA's diffusion block and LLaDA2.2's
# routing block elsewhere): Mima's Rotary(head_dim, block_size) table and generate's crop to block_size
# (modeling_mima.py:140, :202, :313; class default 512, configuration_mima.py:10)
CTX_TYPE_KEYS = {"mima": "block_size"}
_CLASS_DEFAULT_FALLBACK["block_size"] = _R3Dict({}, add={"max_context_unknown": {"mima": 512}})


def _max_context(d, outer=None):
    """(max context, source) of an LM config d: source names the fallback that supplied it, in order --
    'pos_key' (CTX_POS_KEYS, the LM config), 'rope_scaling' (original_max_position_embeddings x factor > 1,
    CTX_ROPE_EXTEND), 'alt_key' (CTX_ALT_KEYS / CTX_TYPE_KEYS, the LM config then the outer wrapper), 'class_default'
    (_class_default: LlamaConfig / FalconConfig 2048, Gemma3TextConfig 131072, Mima's 512) -- else (None, None):
    unknown (dims_of then calls a point with no full-attention layer and no learned table unbounded)."""
    mt = str(d.get("model_type") or "").lower()
    srcs = [d] + ([outer] if isinstance(outer, dict) and outer is not d else [])
    pos = lambda v: _num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0
    v = next((pos(d.get(k)) for k in CTX_POS_KEYS if pos(d.get(k))), 0)
    if v:
        return min(v, 1 << 20), "pos_key"
    rs = next((s[k] for s in srcs for k in ("rope_scaling", "rope_parameters")
               if isinstance(s.get(k), dict) and s[k].get("factor") is not None), None)
    if rs is not None and str(rs.get("rope_type") or rs.get("type") or "") in CTX_ROPE_EXTEND:
        orig = pos(rs.get("original_max_position_embeddings")) or pos(d.get("original_max_position_embeddings"))
        fac = rs.get("factor")
        if orig and isinstance(fac, (int, float)) and not isinstance(fac, bool) and fac > 1:
            return min(int(orig * fac), 1 << 20), "rope_scaling"
    keys = CTX_ALT_KEYS[:-1] + ((CTX_TYPE_KEYS[mt],) if mt in CTX_TYPE_KEYS else ()) + CTX_ALT_KEYS[-1:]
    v = next((pos(s.get(k)) for s in srcs for k in keys if pos(s.get(k))), 0)
    if v:
        return min(v, 1 << 20), "alt_key"
    for k in CTX_POS_KEYS + keys:
        try:
            v = _num(_class_default(mt, k))
        except Exception:
            v = 0
        if v > 0:
            return min(v, 1 << 20), "class_default"
    return None, None


def _ctx_codes(P, k):
    """dims k (max_context / eff_window) of the points as int64 codes: the read value, CTX_UNBOUNDED, or CTX_UNKNOWN for
    None -- an explicit code, never a default (round 3, max_context_unknown)."""
    import numpy as np
    return np.array([CTX_UNKNOWN if p["dims"].get(k) is None else int(p["dims"][k]) for p in P], dtype=np.int64)


def _ctx_ge(h, t):
    """Host context code h supplies target code t: an unbounded host supplies anything; two read values compare; an
    unknown host supplies only an unknown target -- unknown is its own value, so a point with no source still emulates
    itself and the order stays a preorder (transitive, reflexive) -- and an unknown target is supplied only by an
    unbounded host or an unknown one (the conservative reading: its demand may exceed any read value)."""
    return ((h == CTX_UNBOUNDED) | ((h >= 0) & (t >= 0) & (h >= t))
            | ((h == CTX_UNKNOWN) & (t == CTX_UNKNOWN)))


def dims_of(d):
    raw, d = d, _lm_config(d)
    g = lambda *keys: next((_num(d[k]) for k in keys if d.get(k) not in (None, [], "")), 0)
    hidden = g("hidden_size", "n_embd", "n_embed", "d_model", "hidden_dim",
               "model_dim", "dim", "embed_dim")
    n_heads = g("num_attention_heads", "n_head")
    head_dim = ((g("head_dim") if _stock_arch(d) not in _DERIVED_HEAD_DIM_ARCHS else 0)
                or (hidden // n_heads if n_heads else 0))
    n_kv = g("num_key_value_heads", "num_kv_heads", "n_head_kv", "num_key_value_heads_per_layer",
             "num_attention_groups") or _nkv_alias(d) or (1 if d.get("multi_query") else n_heads)
    # kv_cap = per-token KV representational dimension (what an emulator must hold).
    # All KV variants are mutually emulable when kv_cap is large enough (wide shared
    # head / full-rank latent). MLA deliberately compresses -> SMALL kv_cap.
    if d.get("kv_lora_rank"):
        kv_cap = _num(d["kv_lora_rank"]) + g("qk_rope_head_dim")
        q_head = g("qk_nope_head_dim") + g("qk_rope_head_dim")
        q_cap = n_heads * (q_head or head_dim)      # MLA query = nope + rope per head
    else:
        kv_cap = n_kv * head_dim
        q_cap = n_heads * head_dim
    qk_dim = q_cap // max(n_heads, 1)                  # Q/K per-head dim (pattern rank)
    q_heads = n_heads
    pl = None if d.get("kv_lora_rank") else _per_layer_heads(d, n_heads, n_kv, head_dim)
    if pl:                       # round 3 (per_layer_head_shapes): the widest layer of a per-layer list (Laguna-S 72 q
        q_cap = max(q * h for q, _, h in pl)            # heads, Domyn-Edge's 256-wide full layers)
        kv_cap = max(k * h for _, k, h in pl)
        q_heads, qk_dim = max(q for q, _, _ in pl), max(h for _, _, h in pl)
    if _moa(d) and n_kv:
        # round 3: JetMoE's 32 query heads are top_k = 2 routed blocks of num_key_value_heads (configuration_jetmoe.py:
        # 58-59, :73; modular_jetmoe.py:212-282): a MoA HOST ties its experts and offers one block -- n_kv query heads
        # on n_kv KV heads (MoA targets are walled from every non-MoA host: FEATURES moa); kv slot MHA (classify)
        q_heads, q_cap = n_kv, n_kv * head_dim
    # the same capacity over the layers with an INDEPENDENT V only (a K = V layer cannot play a target
    # layer whose V differs from its K, so a K = V host offers only these to such layers)
    kv_cap_u, q_cap_u, qk_dim_u, q_heads_u = kv_cap, q_cap, qk_dim, q_heads
    for vh, vhd, vkv, tied in _attn_variants(d, tagged=True):   # heterogeneous layers: hold the widest
        kv_cap, q_cap = max(kv_cap, vkv), max(q_cap, vh * vhd)
        qk_dim, q_heads = max(qk_dim, vhd), max(q_heads, vh)
        if not tied:
            kv_cap_u, q_cap_u = max(kv_cap_u, vkv), max(q_cap_u, vh * vhd)
            qk_dim_u, q_heads_u = max(qk_dim_u, vhd), max(q_heads_u, vh)
    if _k_eq_v_all(d):                                  # DeepSeek-V4: K = V on every layer, no independent V
        kv_cap_u = q_cap_u = qk_dim_u = q_heads_u = 0
    # FFN capacity. A dense FFN emulates a MoE only by holding ALL experts
    # (ffn_total ~ E*expert_width) plus extra depth for the routing circuit; a MoE
    # emulates a dense FFN with enough ACTIVE width (k*expert_width). So two axes:
    n_exp = 0 if _moe_disabled(d) else (
        g("num_experts", "n_routed_experts", "num_local_experts", "num_routed_experts", "moe_num_experts")
        or _num((d.get("ffn_config") or {}).get("moe_num_experts")))   # Step3 / ERNIE-4.5 / JetMoE / DBRX
    blk_exp = [e for e, _ in _block_experts(d)]               # gpt-oss-Puzzle per-block counts
    if not n_exp and blk_exp:
        n_exp = max(blk_exp)
    hier = _hier_moe(d)
    if hier:
        n_exp = hier[0] * hier[1]               # round 3 (hierarchical_moe_unread): Meditor-2's D x S sub-experts
    if n_exp <= 1:
        n_exp = 0                                             # one expert = a dense MLP (Hymba, mixtral-1x22b)
    k = (g("num_experts_per_tok", "moe_topk", "moe_top_k", "moe_k", *_TOPK_ALT_KEYS)
         or _num((d.get("ffn_config") or {}).get("moe_top_k")))
    if not k and _r3("zoo_deep_topk_legacy") and str(d.get("model_type") or "").lower() == "deep":
        # round 3, the zoo's params triage: DeepConfig takes the legacy top_k key when num_experts_per_tok is absent
        # (configuration_deep.py:52-54; TR-MOE-306, the earlier export with top_k 2 and no code: README "fixed top-2
        # lookup with 0.5/0.5 route weights")
        k = g("top_k")
    if hier:
        k = 1                                   # (one sub-expert per token: two hard argmaxes)
    mt_ = str(d.get("model_type") or "").lower()
    fw3 = _r3("ffn_width_keys_unread")
    dense_inter = (g("intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size", "ff_intermediate_size",
                     *(("ffn_hidden",) if fw3 else ()))
                   or (_ffn_width_by_code(d, mt_, hidden) if fw3 else 0)
                   or (4 * hidden if hidden else 0))
    if _r3("hadamard_mlp_unlisted") and mt_ in HADAMARD_FFN_TYPES and hidden:
        dense_inter = 1 << (int(hidden) - 1).bit_length()     # HadamardMLP's n = next_pow2(d_model) (config.py:328-330)
    if d.get("model_type") in FUSED_GLU_HALF and not (d.get("model_type") == "molmo" and _moe_like(d)):
        # one fused projection of width intermediate_size, split into gate and up halves: the GLU
        # width is intermediate_size // 2 (Qwen v1 w1 / w2, Molmo ff_proj -> SwiGLU chunk)
        dense_inter //= 2
    ffn_sum = None
    ew_r = sh_r = 0                                           # routed expert width, shared width (moe_sig)
    if n_exp and d.get("num_cdmoe_experts"):
        # Doge CDMoE: a dense MLP plus PEER-style single-neuron experts (width 1 each)
        ffn_active, ffn_total = (k or 1) + dense_inter, n_exp + dense_inter
        ew_r, sh_r = 1, dense_inter
    elif n_exp and isinstance(d.get("size_experts"), list) and d["size_experts"]:
        # LLaMA-MoE: the Llama FFN split into experts, widths per layer in size_experts
        per_layer = [[_num(w) for w in row] if isinstance(row, list) else [_num(row)] for row in d["size_experts"]]
        ew = max(max(r) for r in per_layer)
        ffn_total, ffn_active = max(sum(r) for r in per_layer), (k or 1) * ew
        ffn_sum = sum(sum(r) for r in per_layer)
        ew_r = ew
    elif n_exp:
        ew = g("moe_intermediate_size", "moe_ffn_hidden_size", "expert_ffn_hidden_size", "expert_hidden_dim",
               "ffn_hidden_size", "routed_intermediate_size",
               *(("expert_intermediate_size",) if fw3 else ())) or dense_inter
        if fw3 and mt_ == "zaya1_vl" and d.get("gated_linear_unit") and g("ffn_hidden_size"):
            # round 3, batch 3b (ffn_width_keys_unread): ZAYA1-VL's GLU experts fc1 hidden -> ffn_hidden_size, fc2 reads
            # ffn_hidden_size // 2 (modeling_zaya1_vl.py:1715-1730): the GLU branch is half of it
            ew = g("ffn_hidden_size") // 2
        if hier:
            ew = hier[2]                                      # round 3: Meditor-2 sub_intermediate_size
        if _r3("hash_route_by_code") and d.get("model_type") == "deep":
            # round 3 (hash_route_by_code): DeepConfig expert_width = intermediate_size // num_experts
            # (configuration_deep.py:107), not intermediate_size
            ew = max(1, dense_inter // n_exp)
        # always-on shared expert; Nemotron-H states its width explicitly (n_shared_experts=1 there
        # is a count, the shared MLP is moe_shared_expert_intermediate_size wide, 2x an expert)
        shared = (g("moe_shared_expert_intermediate_size") or (g("n_shared_experts") * ew)
                  or g("shared_expert_intermediate_size") or g("shared_intermediate_size")
                  or g("share_expert_dim")                              # Step3
                  or g("moe_num_shared_experts") * ew                   # ERNIE-4.5
                  or g("num_shared_experts") * ew                       # Kimi-Linear / AFMoE / EXAONE-MoE / Hy3 / Ling / ...
                  or g("num_shared_expert") * dense_inter)              # Hunyuan-v1 MoE: shared_mlp, intermediate x count
        if not _has_shared_expert(d):
            shared = 0                  # a template shared width with no shared expert (CENO: n_shared_experts = 0)
        elif _llama4_shared(d):
            shared = g("intermediate_size")     # round 3: Llama4TextMoe.shared_expert = Llama4TextMLP(config)
        if _dense_mlp_beside_moe(d):
            shared = dense_inter        # Gemma-4 / DiffusionGemma: a per-layer dense MLP runs beside the MoE block
        ffn_active, ffn_total = (k or 1) * ew + shared, n_exp * ew + shared
        if blk_exp:
            ffn_sum = sum(e * ew + shared for e in blk_exp)   # whole-model units, per block
        bc = d.get("block_configs")
        if isinstance(bc, list) and any(isinstance(b, dict) and b.get("block_type") == "moe" for b in bc):
            # Nemotron-H-Puzzle: expert width / top-k / expert count set PER MoE BLOCK; the
            # top-level intermediate_size is the dense template, not an expert width
            per = [((_num(b.get("n_routed_experts")) or n_exp),
                    (_num(b.get("num_experts_per_tok")) or k or 1),
                    (_num(b.get("moe_intermediate_size")) or ew))
                   for b in bc if isinstance(b, dict) and b.get("block_type") == "moe"]
            ffn_total = max(e * w for e, _, w in per) + shared
            ffn_active = max(kk * w for _, kk, w in per) + shared
            k = max(kk for _, kk, _ in per)                  # the per-block top-k (the top level is a template)
            ffn_sum = sum(e * w + shared for e, _, w in per)
            ew = max(w for _, _, w in per)
        ew_r, sh_r = ew, shared
    else:
        ffn_active = ffn_total = dense_inter
    if mt_ in FFN_NONE_TYPES:
        # round 3 (mamba_ffn_none): no FFN, no width (intermediate_size is the mixer's d_inner: _rec_dims rc_h)
        ffn_active = ffn_total = dense_inter = 0
    # per-layer FFN widths where layers differ (OpenELM ffn_multipliers, an intermediate_size list; Nemotron-NAS
    # below): each target layer needs a host layer at least as wide (emul_matrix: _ffn_widths_fit), so the widest
    # layer alone is not the host's width on every layer
    ffn_widths = ()
    iw = d.get("intermediate_size")
    if not n_exp and isinstance(iw, list) and len({_num(x) for x in iw}) > 1:
        ffn_widths = tuple(sorted((_num(x) for x in iw), reverse=True))
        ffn_sum = sum(ffn_widths)
    # depth: param_layers = weight-DISTINCT layers, eff_layers = unrolled forward depth.
    # For a plain stack they coincide; a looped/recurrent-depth model has few unique
    # layers (its block) but a deep unrolled forward pass (block x recurrence).
    n_layers, param_layers, eff_layers = _loop_layers(d)
    lt = d.get("layer_types")
    if (isinstance(lt, list) and "mlp" in lt and not any(_rec_layer(x) for x in lt)
            and len(lt) == n_layers):
        # Dory: the layer list names attention and MLP SUB-layers separately ('sliding_attention',
        # 'mlp', ...); one attention + FFN block is one layer of depth, so count the attention entries
        frac = sum(1 for x in lt if str(x) != "mlp") / len(lt)
        n_layers = max(1, round(n_layers * frac))
        param_layers, eff_layers = max(1, round(param_layers * frac)), max(1, round(eff_layers * frac))
    # Nemotron-NAS (DeciLM) stores attention / FFN shapes PER BLOCK: GQA via n_heads_in_group,
    # no-op attention or FFN in many blocks, a different FFN width per block.
    bc = d.get("block_configs")
    if d.get("model_type") == "nemotron-nas" and isinstance(bc, list) and bc:
        att = [b.get("attention", {}) for b in bc]
        groups = [a["n_heads_in_group"] for a in att if not a.get("no_op") and a.get("n_heads_in_group")]
        if groups:
            kv_cap = kv_cap_u = (n_heads // min(groups)) * head_dim   # widest per-block KV cache
            n_kv = n_heads // min(groups)
        inters = []
        for b in bc:
            f = b.get("ffn", {})
            if not f.get("no_op") and f.get("ffn_mult"):
                inter = int(2 * f["ffn_mult"] * hidden / 3)
                inters.append(-(-inter // 256) * 256)               # DeciLM rounds up to 256
        if inters:
            ffn_active = ffn_total = max(inters)
            ffn_sum = sum(inters)                                   # whole-model FFN units
            ffn_widths = tuple(sorted(inters, reverse=True))        # each live FFN block's own width
        live = sum(1 for b in bc if not (b.get("attention", {}).get("no_op") and b.get("ffn", {}).get("no_op")))
        param_layers = eff_layers = live or n_layers
    # attention reach: max context supported, and per-layer receptive field (sliding window).
    mpe = g("max_position_embeddings", "max_seq_len", "n_positions", "seq_length")
    if _r3("max_context_unknown"):
        # round 3 (max_context_unknown): the fallbacks of _max_context (the position keys, rope orig x factor, the alt
        # length keys -- Kimi-Linear's model_max_length 1048576 --, the class default), else None = unknown; a
        # recurrent / window-only point is unbounded below, once its layer kinds are known. No 4096 default
        max_context = _max_context(d, raw if isinstance(raw, dict) else None)[0]
    else:
        max_context = min(int(mpe), 1 << 20) if mpe else 4096      # clamp 10M+ placeholders; default if missing
    sw = _num(d.get("sliding_window")) if (d.get("sliding_window") and d.get("use_sliding_window", True)) else 0
    if (_r3("max_context_unknown") and not sw and _num(d.get("attention_window_size"))
            and isinstance(d.get("_block_types") or d.get("block_types"), list)):
        # round 3 (max_context_unknown): RecurrentGemma's attention layers are local, attention_window_size wide (the
        # ring-buffer cache, modeling_recurrent_gemma.py:216-246; _attn_counts already calls them local): that is
        # their reach, not the full context
        sw = _num(d["attention_window_size"])
    eff_window = int(sw) if sw else max_context                # sliding window, else full-context reach
    ch = _num(d.get("attention_chunk_size"))
    if (_r3("chunked_attention_unread") and not sw and ch and mpe and ch < mpe
            and str(d.get("model_type") or "").lower() in ("llama4", "llama4_text")
            and any(_num(x) for x in (d.get("no_rope_layers") or []))):
        # round 3: Llama-4's RoPE layers are 'chunked_attention' (Llama4TextConfig.__post_init__, tf5
        # configuration_llama4.py:176-202): causal AND the same attention_chunk_size block (masking_utils.py:106-113,
        # :1402-1404) -- block-diagonal with absolute boundaries, not a band. The kinds stay FU (a full host layer
        # plays the chunk mask, pattern_sub_full; a Llama-4 host sets attention_chunk_size >= T, a window size
        # transformers' create_chunked_causal_mask reads -- the window lever's reading, window_ge_full's proof shape;
        # _kind would call a 'chunked' layer local, so no layer_types are derived). Only the reach moves (context axis)
        eff_window = min(eff_window, ch)
    # whole-model FFN units = the FFN actually present in each layer, summed. A MoE stack whose
    # first k layers (first_k_dense_replace) or whose mlp_layer_types 'dense' entries are a plain
    # MLP holds dense_inter there, not the full expert bank.
    # round 3, batch 3b (ffn_width_keys_unread): a MoE stack's dense layers are dense_intermediate_size wide where the
    # code builds them so (MiniMax-M3 MiniMaxM3VLDenseMLP(intermediate_size=config.dense_intermediate_size),
    # modular_minimax_m3_vl.py:629; IQuest-Q1's mlp_only_layers IQuestQ1DenseMLP, model.py:451-453), not
    # intermediate_size (the expert width there)
    dense_w = (g("dense_intermediate_size") if n_exp and fw3 else 0) or dense_inter
    sub = _ffn_sublayers(d) if ffn_sum is None and not _looped(d) else None
    if sub:
        ffn_units = sub[0] * ffn_total + sub[1] * dense_w
    elif ffn_sum is not None:
        ffn_units = ffn_sum
    elif n_exp and not _looped(d) and any(_moe_layer_mask(d, n_layers) or ()):
        mask = _moe_layer_mask(d, n_layers)          # the per-layer dense / MoE kinds (round 2, pass 3)
        ffn_units = sum(mask) * ffn_total + (len(mask) - sum(mask)) * dense_w
    elif not n_exp and d.get("use_double_wide_mlp") and 0 < _num(d.get("num_kv_shared_layers")) < n_layers:
        # Gemma-4 E-models: the KV-shared (last num_kv_shared_layers) layers' MLP is 2x intermediate_size
        n_wide = _num(d["num_kv_shared_layers"])
        ffn_total = ffn_active = 2 * dense_inter
        ffn_units = (n_layers - n_wide) * dense_inter + n_wide * 2 * dense_inter
    else:
        ffn_units = eff_layers * ffn_total
    # ZERO_EDGES depth: the layers a plain softmax target can use (own-KV attention; full window)
    n_list = (g("num_hidden_layers", "n_layer", "num_layers", "num_transformer_layers", "n_layers")
              or _layer_count(d) or (n_layers if _r3("loop_depth_unread") else 0))   # (round 3: CFRD's cells)
    a_cnt, f_cnt, base, shared_cap, kc, ktoks = _attn_counts(d, n_list)
    scale = lambda c, tot: min(tot, round(c * tot / base))
    attn_eff, full_eff = scale(a_cnt, eff_layers), scale(f_cnt, eff_layers)
    attn_param, full_param = scale(a_cnt, param_layers), scale(f_cnt, param_layers)
    kind_dims = {}
    for kind in LAYER_KINDS:                             # per-layer-kind counts (LAYER_KINDS), unrolled / distinct
        kind_dims[f"k{kind}_eff"] = scale(kc[kind], eff_layers)       # (not `k`: that is the router top-k, read by
        kind_dims[f"k{kind}_param"] = scale(kc[kind], param_layers)   #  moe_unit_top1 below)
    if shared_cap:                                       # one block shared across many positions
        attn_param, full_param = min(attn_param, shared_cap), min(full_param, shared_cap)
        for kind in LAYER_KINDS:
            if kind != "R":
                kind_dims[f"k{kind}_param"] = min(kind_dims[f"k{kind}_param"], shared_cap)
    # round 3 (kind_order): the unrolled layer-kind SEQUENCE ("" where it is one plain kind: the counts say it all)
    kind_dims["kseq"] = _kind_seq(ktoks, base, eff_layers) if _r3("kind_order") else ""
    if _r3("d_transitivity_depth"):
        # round 3 (d_transitivity_depth): the zero-edge counts ARE the own-KV kind counts (one rounding of one layer
        # list): a plain target charges attn_eff / full_eff, a non-plain one the kinds (kind_match), and a point whose
        # attn_eff fell below its own-KV kind total broke depth transitivity (verify/z3_relation.py's "built" wf clause)
        own = ("FU", "FT", "LU", "LT")
        attn_eff = sum(kind_dims[f"k{x}_eff"] for x in own)
        attn_param = sum(kind_dims[f"k{x}_param"] for x in own)
        full_eff = kind_dims["kFU_eff"] + kind_dims["kFT_eff"]
        full_param = kind_dims["kFU_param"] + kind_dims["kFT_param"]
    # fr-positions (fp_nope_layer_host): the ROTATING attention layers per kind (rk*): a RoPE + NoPE mix's RoPE layers
    # (_nope_flags), every attention layer elsewhere; a mix whose NoPE layers cannot be located rotates none (the old
    # wall: it hosts no rotating target layer). emul_matrix's rotating-layer Hall reads them
    if _r3("fp_nope_layer_host"):
        nf = _nope_flags(d, n_list) if _nope_layer_mix(d) else None
        rkc = _attn_counts(d, n_list, only=[not x for x in nf]) if nf is not None else None
        # (release, with round 3's kind_order: the rotating layers' kind SEQUENCE too -- the NoPE layers dropped, the
        # rest in order -- so the rotating-layer check is an order-preserving embedding into the host's RoPE layers,
        # not a count beside a separate order check of all layers: a count and an order check passing apart do not
        # give ONE embedding that is both. An embedding of embeddings is one, so it stays transitive)
        kind_dims["rkseq"] = kind_dims["kseq"]
        if rkc is not None and _r3("kind_order"):
            kind_dims["rkseq"] = _kind_seq(rkc[5], base, eff_layers)
        rkc = rkc[4] if rkc is not None else None
        if nf is None and _nope_layer_mix(d):
            nf = ()                                        # a mix whose NoPE layers cannot be located
            kind_dims["rkseq"] = ""                        # (rotates none: the counts, all 0, decide)
        for kind in ("FU", "FT", "LU", "LT"):
            for w, tot in (("eff", eff_layers), ("param", param_layers)):
                if nf is None:
                    v = kind_dims[f"k{kind}_{w}"]
                elif rkc is None:
                    v = 0
                else:
                    v = min(kind_dims[f"k{kind}_{w}"], scale(rkc[kind], tot))
                    if w == "param" and shared_cap:
                        v = min(v, shared_cap)
                kind_dims[f"rk{kind}_{w}"] = v
    sched = _rope_schedules(d, raw if isinstance(raw, dict) else None)
    p5 = _pass5_dims(d, raw if isinstance(raw, dict) else None)
    if (_r3("max_context_unknown") and max_context is None and not p5["pos_table"]
            and sum(kind_dims[f"k{x}_eff"] for x in ("S", "R", "LU", "LT")) > 0
            and kind_dims["kFU_eff"] + kind_dims["kFT_eff"] == 0):
        # round 3 (max_context_unknown): a recurrent or window-only point (no full-attention layer, no learned table)
        # with no length in its config or class has no limit in its code (Mamba / Falcon-Mamba: the scan or the
        # recurrent step, any length; RecurrentGemma: the RG-LRU plus window-local attention over a ring buffer, the
        # mask sized max(window, T), modeling_recurrent_gemma.py:661): unbounded (arch_metrics' rule; it checks this
        # before the alt keys, here a read length -- RWKV's context_length, the CUDA kernel's Tmax -- wins). Its layers'
        # reach is the window where it has one
        max_context = CTX_UNBOUNDED
        eff_window = eff_window if eff_window is not None else CTX_UNBOUNDED
    moe_sig = _moe_sig(d, n_exp, k, ew_r, sh_r, ffn_total if not n_exp else dense_w, hidden, n_layers,
                       eff_layers, outer=raw if isinstance(raw, dict) else None)
    if n_exp and _moe_fields(moe_sig)["par_res"]:
        # round 3 (arctic_dense_residual_parallel_moe): Arctic's residual MLP (width hidden_size) on every MoE layer
        ffn_units += _moe_fields(moe_sig)["n_moe"] * _moe_fields(moe_sig)["par_res"]
    out = {
        # RoPE (read only where the pos slot is rope / partial_rope; _rope_schedules): per distinct
        # frequency schedule (one inv_freq table each), widest first -- rope_sched = the rotated dims per
        # head a TARGET uses, rope_cap = the rotated dims per head a HOST's table can drive (rope_fits)
        "rope_sched": tuple(sorted((n for n, _ in sched), reverse=True)),
        "rope_cap": tuple(sorted((c for _, c in sched), reverse=True)),
        "hidden": hidden,
        "layers": n_layers,
        "heads": n_heads,
        "inter": dense_inter or g("moe_intermediate_size"),
        "experts": n_exp,
        "kv_cap": kv_cap,
        "q_cap": q_cap,   # query-side resolution (width)
        "qheads": q_heads,  # query-head COUNT (richness: number of distinct softmaxes)
        "head_dim": qk_dim,  # Q/K per-head dim (widest layer) = attention-pattern RANK bound
        "ffn_active": ffn_active,
        "ffn_total": ffn_total,
        "param_layers": param_layers,  # weight-distinct layers (looping trades these away)
        "ffn_units": ffn_units,        # whole-model FFN units (only the sub-layers that carry an FFN)
        "eff_layers": eff_layers,      # unrolled forward depth (block x recurrence)
        "vocab": g("vocab_size", "n_vocab", "padded_vocab_size"),  # embedding/head capacity (off by default)
        "max_context": max_context,   # max sequence length supported (a DEMAND currency; off by default)
        # round 3, batch 2b (PLAN decision 3): the position where a context gate starts (0 = none; _pos_gate): the
        # credited context is capped there under USE_CONTEXT
        "pos_gate": _pos_gate(d, raw if isinstance(raw, dict) else None),
        "eff_window": eff_window,     # per-layer attention reach (sliding window, else full context)
        "attn_eff": attn_eff,         # own-KV softmax-attention layers (any window), unrolled
        "attn_param": attn_param,     #   ... weight-distinct
        "full_eff": full_eff,         # of them with a FULL window, unrolled
        "full_param": full_param,     #   ... weight-distinct
        # the capacity of the layers with an independent V (= the above unless the model has K = V layers)
        "kv_cap_u": kv_cap_u, "q_cap_u": q_cap_u, "qheads_u": q_heads_u, "head_dim_u": qk_dim_u,
        # MLA latents (0 = none): the KV latent rank, the query latent rank (q_lora_rank: a second
        # per-token latent norm), the shared rotary key width
        # KV heads and query heads per KV group: each target KV head needs its own host KV head(s), each with
        # enough query heads (MLA: every head has its own K / V from the latent)
        # a MoE whose dense emulation is a hard select: top-1 with a unit (renormalised) gate
        "moe_unit_top1": int(bool(n_exp) and moe_sig[1] == 1 and moe_sig[4] == "R"),
        # the routing facts the MoE rules compare (MOE_SIG, moe_why): experts, top-k, widths, renorm / score /
        # group-score (each with what the family's code or config can be set to), dense / MoE layer counts
        "moe_sig": moe_sig,
        "kv_heads": n_heads if d.get("kv_lora_rank") else n_kv,
        "q_per_kv": 1 if d.get("kv_lora_rank") else max(1, n_heads // max(n_kv, 1)),
        "kv_latent": _num(d.get("kv_lora_rank")), "q_latent": _num(d.get("q_lora_rank")) if d.get("kv_lora_rank") else 0,
        "mla_rope": g("qk_rope_head_dim") if d.get("kv_lora_rank") else 0,
        # MLA per-head widths, compared on their own (round 2, pass 3): the latent-sourced nope part of q / k and V
        "mla_nope": g("qk_nope_head_dim") if d.get("kv_lora_rank") else 0,
        "mla_v": (g("v_head_dim") or g("qk_nope_head_dim")) if d.get("kv_lora_rank") else 0,
        # per-layer FFN widths, widest first, where the layers differ (() = every layer ffn_total wide)
        "ffn_widths": ffn_widths,
        **kind_dims,                  # kFU_eff, kFU_param, ... (LAYER_KINDS)
        **_alibi_dims(d),             # alibi_heads / alibi_max / alibi_bf16 / alibi_off: the slope scheme
        # 1 = the rotary takes its rope type from the config (longrope: a frequency per pair), so a table may
        # repeat frequencies (KV-head packing of a RoPE target, kv_heads_fit)
        "rope_pairs_free": int(str(d.get("model_type") or "").lower() in ROPE_PAIRS_FREE_TYPES),
        # round 3 (nope_rope_limit): 2 / 1 = the rotary reads linear rope scaling per table / for its one rotating
        # table (a host config lever: factor -> inf, the NoPE limit), 0 = it does not (_rope_lin)
        "rope_lin": _rope_lin(d, sched),
        # recurrent capacity (round 2, pass 4: RC_DIMS / rc_rule, _rec_ok) and the Zamba shared blocks
        **_rec_dims(d),
        **_zamba_dims(d),
        # round 2, pass 5: the host's config levers and the start-token / rescue facts (_pass5_dims); a learned-abs
        # table's length (pos_table: the start-token rule's T + 1 = it + the marker)
        **p5,
        # (round 3, max_context_unknown: a table of unknown length is 0 -- the start-token rule then pays nothing)
        "pos_table": (max_context or 0) if p5["pos_table"] else 0,
        # fr-positions (fp_sinusoid_table): a FIXED sinusoid table's identity -- the code that computes it and its width
        # (XGLMSinusoidalPositionalEmbedding / CTRL positional_encoding / Musicgen's: a function of the position, the
        # model type's formula and embedding_dim = hidden only), so two sinusoidal points hold the same table iff equal
        "sin_sig": (f"{str(d.get('model_type') or '').lower()}:{hidden}"
                    if (_r3("fp_sinusoid_table") and classify(d) and classify(d)["pos"] == "sinusoidal") else ""),
        # the universal host's MLA branches (minimal_emulator.host_point); 0 on every real model
        "mla_qswitch": 0, "mla_lat_free": 0,
        # per-layer attention shapes beyond the top-level one, (kv heads, q per kv, head dim) each (round 2, pass 5,
        # verify/REVIEW.md: MiMo-V2.5's sliding layers 8 KV heads x 8 q vs the top-level 4 x 16; kv_heads_fit)
        "kv_shapes": _kv_shapes(d, n_heads, n_kv, head_dim),
        # round 3, batch 1: embedding factorization / norm, grouped RMSNorm, gated pre-norm (_r3_dims)
        **_r3_dims(d),
    }
    out.update(_r3_dims_2d(d, raw if isinstance(raw, dict) else None, out, n_heads, n_kv, head_dim))
    # round 3, batch 3b: the FFN up-projection bias (bias_up: spot_vanilla_glu_bias, w3_relu_oai_bias) and needle's
    # HadamardMLP width (hada_n: hadamard_mlp_unlisted; 0 = not a HadamardMLP)
    out["bias_up"] = out["ffn_bias"]
    out["hada_n"] = int(dense_inter) if (_r3("hadamard_mlp_unlisted")
                                         and str(d.get("model_type") or "").lower() in HADAMARD_FFN_TYPES) else 0
    out.update(_r3_dims_4b(d, raw if isinstance(raw, dict) and raw is not d else None))
    out.update(_readout(d, raw if isinstance(raw, dict) and raw is not d else None)[2])     # round 3, nonlm_heads
    return out


# round 3 (nonlm_heads) dims: the readout of a non-LM head (_readout; neutral "" / 0 for an LM), compared by
# _r3_masks_nonlm between two heads of one kind:
#   readout_sig   an embedding's pooling + L2 ("last+l2", "mean", "mean+l2"; FIXED_READS "last" / "first" are a fixed
#                 read of an LM's output)
#   readout_w     the head's output width (score / value 1, an embedding's Dense out or hidden, the flow target width)
#   readout_ff    the value head's hidden width (GRM vhead_num_neurons)
#   readout_bias  the head has a bias (stella's Dense, GRM's ValueHead)
#   flow_sig / patch_sig  sundial's flow net (target width / width / adaLN depth) and patch embedding (patch / MLP)
#   exits         the score heads' exit layers (informational: the default call reads one, at L; the join takes the max)
R3_DIMS_NONLM = ("readout_sig", "readout_w", "readout_ff", "readout_bias", "flow_sig", "patch_sig", "exits")
# round 3 (nope_rope_limit): the linear-scaling host lever (dims_of rope_lin, _rope_lin; 0 everywhere with it off)
R3_DIMS_NOPE = ("rope_lin",)


# round 3, batch 4b dims (compared by _r3_masks_4b; neutral "" / 0 elsewhere):
#   attn_res_sig  the AttnRes source partition (_attn_res_sig; attn_res_keys_unread)
#   engram_sig    the n-gram memory's table geometry (_engram_kind; qwen4_ple_reads_ngrams / ngram_memory_unread)
#   ve_kind       the value-embedding kind (_ve_kind; value_emb_keys_unread)
#   bias_shift    1 = the bias_qkv / bias_out of the config are ONE norm shift shared by every consumer (CFRD
#                 phase_rms_norm; cfrd_film_norm_shift): no independent biases to play a biased target with
R3_DIMS_4B = ("attn_res_sig", "engram_sig", "ve_kind", "bias_shift")


def _r3_dims_4b(d, outer=None):
    f = _feats(d, outer)
    mt = str(d.get("model_type") or "").lower()
    return {"attn_res_sig": _attn_res_sig(d, f), "engram_sig": _engram_kind(d)[1], "ve_kind": _ve_kind(d, f),
            "bias_shift": int(_r3("cfrd_film_norm_shift") and mt == "cfrd")}


# round 3, batch 3b dims: bias_up = the FFN's gate / up INPUT projection carries a bias (the facts of _ffn_bias: an
# explicit mlp_bias / use_bias / add_bias_linear / ... flag, else _GATE_BIAS_TYPES; for a MoE host the experts' up
# bias, gpt-oss gate_up_proj_bias) -- the named predicate the vanilla <- GLU / MoE step (EP.glu_ungated: up weights 0,
# bias 1) and ReLU <- swigluoai (Lean relu_sub_swigluoai: up bias -1) need; hada_n = HadamardMLP's n
R3_DIMS_3B = ("bias_up", "hada_n")


R3_DIMS_2D =("v_head", "kv_shape_n", "o_groups", "o_rank", "q_lat_norm", "canon", "moa_e", "moa_k")


def _r3_dims_2d(d, outer, dm, n_heads, n_kv, head_dim):
    """Round 3, batch 2d dims of an LM config d (neutral values elsewhere), and the corrections of dims_of's dm:
      v_head      the V width per KV head of the top-level shape (v_head_dim_narrow; = head_dim elsewhere)
      kv_shape_n  the attention layers of each kv_shapes entry, summing to the FU + FT + LU + LT eff count
                  (per_layer_head_shapes; () for single-shape stacks: kv_heads_fit counts their attention layers)
      o_groups / o_rank  DeepSeek-V4 / V4.1 / MemT / ERNIE-5.5's grouped low-rank o_proj, o_groups and o_lora_rank
                  (grouped_lowrank_oproj; 0 = a dense o_proj)
      q_lat_norm  Step-3's normed query latent share_q_dim (step3_mfa_query_latent_norm; 0 = none)
      canon       the Canon conv positions / kernel "<ABCD>/k<kernel>" (canon_conv_layers; "" = none)
      moa_e / moa_k  JetMoE's attention experts and top-k (moa_routed_attention_heads; 0 = no MoA)
    and the ERNIE-5.5 MLA widths from its hybrid_mla_* keys (ernie_mla_layer_dims_unread)."""
    mt = str(d.get("model_type") or "").lower()
    out = {}
    # KV shapes, V widths and per-shape layer counts
    tab = _kv_shape_table(d, n_heads, n_kv, head_dim)
    shapes = tuple(sorted(tab)) if len(tab) > 1 else ()
    out["kv_shapes"] = shapes
    total = sum(int(dm.get(f"k{k}_eff", 0)) for k in ("FU", "FT", "LU", "LT"))
    out["kv_shape_n"] = _kv_shape_counts(tab, shapes, total)
    out["v_head"] = _v_head(d, dm["head_dim"]) if _r3("v_head_dim_narrow") else int(dm["head_dim"])
    if _moa(d) and n_kv:
        out["q_per_kv"] = 1                              # (one tied expert block: n_kv q heads on n_kv KV heads)
    em = _ernie_mla(d)
    if em:
        out.update(kv_latent=em["kv_latent"], q_latent=em["q_latent"], mla_rope=em["mla_rope"],
                   mla_nope=em["mla_nope"], mla_v=em["mla_v"])
    og, orank = 0, 0
    if _r3("grouped_lowrank_oproj") and mt in GROUPED_O_TYPES:
        og, orank = _num(d.get("o_groups")), _num(d.get("o_lora_rank"))
        if not (og >= 1 and orank > 0):
            og = orank = 0
    out["o_groups"], out["o_rank"] = og, orank
    out["q_lat_norm"] = ((_num(d.get("share_q_dim")) or 2048)
                         if _r3("step3_mfa_query_latent_norm") and mt in ("step3_text", "step3_vl") else 0)
    out["canon"] = _canon(d, outer) or ""
    moa = _moa(d)
    out["moa_e"], out["moa_k"] = moa or (0, 0)
    return out


# round 3 (grouped_lowrank_oproj): o = W_b blockdiag(A_1 .. A_g) attn, o_groups groups each of rank o_lora_rank
# (tf5 modular_deepseek_v4.py:278-307, :697-700, :775-777; V4.1 inference/model.py:71-76, :634; MemT; ERNIE-5.5
# vLLM-ERNIE attention.py:176-185, :230-243)
GROUPED_O_TYPES = ("deepseek_v4", "deepseek_v41", "deepseek_v41_text", "memt_deepseek_v4", "ernie5_5_moe")


def _canon(d, outer=None):
    """Round 3 (canon_conv_layers): the Canon layers' "<positions>/k<kernel>" -- residual causal depthwise convs at A
    (attention input), B (q / k / v), C (MLP input), D (MLP hidden): dhara_ar canon_set / canon_kernel
    (modeling_dhara_ar.py:46-100, :276-300, :417-471), Ethoric canon {enabled, positions, kernel}
    (modeling_ethoric.py:38-74, :126-159), Needle extras.qkv_conv_taps (cactus architecture.py:184-203: B) -- else None."""
    if not _r3("canon_conv_layers"):
        return None
    mt = str(d.get("model_type") or "").lower()
    outer = outer or {}
    if mt == "dhara_ar":
        cs = d.get("canon_set")
        pos = "".join(sorted(set(str(cs).upper()) & set("ABCD"))) if isinstance(cs, str) else ""
        k = _num(d.get("canon_kernel", 4))
        return f"{pos}/k{k}" if pos and k > 1 else None
    if mt == "ethoric":
        c = d.get("canon") if isinstance(d.get("canon"), dict) else (
            outer.get("canon") if isinstance(outer.get("canon"), dict) else {})
        if not c.get("enabled", False):
            return None
        pos = "".join(sorted({str(p).upper() for p in (c.get("positions") or []) if isinstance(p, str)} & set("ABCD")))
        k = _num(c.get("kernel", 4))
        return f"{pos}/k{k}" if pos and k > 1 else None
    if mt == "needle":
        ex = d.get("extras") if isinstance(d.get("extras"), dict) else (
            outer.get("extras") if isinstance(outer.get("extras"), dict) else {})
        k = _num(ex.get("qkv_conv_taps", d.get("qkv_conv_taps", 0)))
        return f"B/k{k}" if k > 1 else None
    return None


def _canon_covers(t, h):
    """A Canon host h ("<pos>/k<k>") for a Canon target t. The spec's cover (every target position, kernel >= the
    target's: the unused convs and the extra taps 0) needs theorem canon_pad and has no catalog instance, so until one
    lands (CANON_PAD) the host's Canon layers must equal the target's (no snapshot pair of Canon points has an edge)."""
    if not t or not h or not CANON_PAD:
        return t == h
    try:
        (pt, kt), (ph, kh) = t.split("/k"), h.split("/k")
        return set(pt) <= set(ph) and int(kh) >= int(kt)
    except ValueError:                                     # (a value no config gives: only itself)
        return t == h


CANON_PAD = False


def _moa(d):
    """Round 3 (moa_routed_attention_heads): JetMoE's MoA (E, top-k) -- num_local_experts / moe_num_experts and
    num_experts_per_tok / moe_top_k, tf5 defaults 8 / 2 (modular_jetmoe.py:106-149, :212-373) -- else None."""
    if not _r3("moa_routed_attention_heads") or str(d.get("model_type") or "").lower() != "jetmoe":
        return None
    e = _num(d.get("num_local_experts", d.get("moe_num_experts", 8)))
    k = _num(d.get("num_experts_per_tok", d.get("moe_top_k", 2)))
    return (int(e), int(k)) if e > 1 and k >= 1 else None


def build_classes(items):
    """items: iterable of (name, config_dict). Groups by slot-tuple."""
    classes = {}
    n_seen = 0
    for name, d in items:
        if _junky(name):
            continue
        n_seen += 1
        c = classify(d)
        if c is None:
            continue
        key = tuple(c[s] for s, _ in SLOTS)
        grp = classes.setdefault(key, {"slots": c, "members": [], "dims": DIM0.copy(),
                                       "biggest": (None, -1)})
        if name not in grp["members"]:
            grp["members"].append(name)
        dm = dims_of(d)
        for kk in grp["dims"]:
            grp["dims"][kk] = max(grp["dims"][kk], dm[kk])
        score = dm["hidden"] * dm["layers"]
        if score > grp["biggest"][1]:
            grp["biggest"] = (name, score)
    # representative label first: prefer real (non-tiny/test) short names
    for grp in classes.values():
        grp["members"].sort(key=lambda n: ("tiny" in n.lower() or "test" in n.lower(),
                                            len(n.split("/")[-1]), n.lower()))
    return classes, n_seen


def short(name):
    return name.split("/")[-1]


def slot_str(s):
    """Core 5 slots + compact flags for all the extra axes."""
    core = f"{s['norm']} | {s['pos']} | {s['kv']} | {s['ffn']} | {s['act']}"
    mx = s.get("mixer", "softmax")
    flags = ([mx] if mx != "softmax" else []) + [
        n for n, on in [("parallel", s.get("block") == "parallel"),
                        ("sliding", s.get("window") == "local"),
                        (f"qk-norm:{s.get('qknorm')}", s.get("qknorm", "no") != "no"),
                        (f"softcap:{s.get('softcap')}", s.get("softcap", "no") != "no"),
                        ("sink", s.get("sink") == "yes"), ("sink:key", s.get("sink") == "key"),
                        ("xlayer-kv", s.get("crosslayer_kv") == "yes"),
                        ("mtp", s.get("mtp") == "yes"),
                        ("diffusion", s.get("gen") == "diffusion"),
                        ("looped", s.get("loop") == "looped")] if on]
    return core + (" | +" + "+".join(flags) if flags else "")


def build_dot(classes):
    keys = list(classes)
    ids = {k: f"n{i}" for i, k in enumerate(keys)}
    # all comparable pairs
    edges = {}
    for a in keys:
        for b in keys:
            if a == b:
                continue
            ok, diffs = le(classes[a]["slots"], classes[b]["slots"])
            if ok:
                edges[(a, b)] = diffs
    # transitive reduction: drop a->b if exists a->c->b
    cover = {}
    for (a, b), diffs in edges.items():
        if any((a, c) in edges and (c, b) in edges for c in keys if c not in (a, b)):
            continue
        cover[(a, b)] = diffs

    COLOR = {"vanilla": "#dfe7f2", "glu": "#d7ead0", "moe": "#f6e2c0", "moe_shared": "#f2cdbb"}
    lines = ["digraph subsets {", '  rankdir=BT; node [shape=box, style="rounded,filled", '
             'fontname="Helvetica", fontsize=10]; edge [fontname="Helvetica", fontsize=8];']
    for k in keys:
        s = classes[k]["slots"]
        mem = classes[k]["members"]
        rep = short(mem[0]) + (f"  (+{len(mem)-1})" if len(mem) > 1 else "")
        label = (f"{rep}\\n{s['norm']} | {s['pos']} | {s['kv']}\\n{s['ffn']} | {s['act']}")
        fill = "#ffe08a" if s["norm"] == "AffineNorm" else COLOR.get(s["ffn"], "#eeeeee")
        lines.append(f'  {ids[k]} [label="{label}", fillcolor="{fill}"];')
    for (a, b), diffs in cover.items():
        cur = ",".join(sorted({c for (_, cost) in diffs if cost for c in cost[0]}))
        exact = all(cost[1] for (_, cost) in diffs if cost)
        style = "solid" if exact else "dashed"
        lines.append(f'  {ids[a]} -> {ids[b]} [label="{cur}", style={style}];')
    lines.append("}")
    return "\n".join(lines), edges, cover


AXES = ("hidden", "layers", "inter", "experts")   # kv/q handled by capacity below
MOE_DEPTH_FACTOR = 2   # depth a dense FFN needs to emulate a MoE (routing circuit); heuristic
# (round 3, batch 3b, spot_glu_vanilla: the factor now prices only routes with a construction -- a dense GLU host for a
# top-1 unit-gate MoE target, the select + expert layer pair of moe/dense_host_prenorm.py, Lean moe_two_layer_margin,
# and a parallel host of a sequential target; a gated / MoE target on a vanilla host is a wall in emul_matrix)
FFN_RANK = {"vanilla": 0, "glu": 1, "moe": 2, "moe_shared": 3}   # FFN "richness"
# round 3, batch 3b (hadamard_mlp_unlisted): needle's HadamardMLP ranks as a dense FFN; it is hosted only by its own code
# at equal n and hosts nothing else (emul_matrix, dims hada_n), so the rank only orders its depth against itself
FFN_RANK["hadamard"] = 1
# round 3 (mamba_ffn_none): no FFN ranks below vanilla (no 2x depth is ever charged for it)
FFN_RANK["none"] = -1
# TransMLA: at equal cache MLA is strictly more expressive than tied-KV variants,
# so a GQA/MQA emulator needs strictly MORE kv_cap to reproduce an MLA target.
MLA_PENALTY = 2
# COST_RELAX (branch fr-cost, graph/fr_cost.py): a COUNTERFACTUAL hook, empty in the relation. A name here sets that
# construction's charge to its floor (multiplier 1, extra 0) WITHOUT crossing any new wall: graph/fr_cost.py ranks the
# charges by the frontier classes a cheaper construction would free. Never set by the relation itself.
COST_RELAX = set()
COST_RELAX_NAMES = ("act_k", "norm_mirror", "par_width", "mla_penalty", "depth2", "bos", "sink", "loop", "kv_heads",
                    "kv_rope_pack", "mla_mult", "rec", "moe_cap", "moe_dense", "moe_topk", "layer_kinds", "window")


def _cr(name):
    """0 if the charge `name` is relaxed (COST_RELAX), else 1."""
    return 0 if name in COST_RELAX else 1


# branch fr-cost (cost_norm_subspace): features whose norm output is written INTO the residual with a per-channel gain
# (sandwich / post-only / post-LN / branch / loop norms, nGPT, ...) or not read off the snapshot as pre-norm only: the
# RMSNorm <- LayerNorm pair keeps the 2x mirror where either side has one. Norms read by a linear map before the
# residual (vnorm, onorm, ffn_norm: o_proj / down_proj write) are fine.
NORM_SUBSPACE_DENY = frozenset({"post_norm", "post_norm_attn", "post_only", "post_ln", "moe_branch_norm", "expert_norm",
                                "loop_bnorm", "loop_bnorm_cfg", "ngpt", "selfcond_norm", "gated_norm", "out_gate_post"})


def _norm_subspace_ok(P, host=False):
    """[i]: point i may sit on the cheap side of the subspace rule (no residual-written gained norm, no embedding norm;
    a host also untied)."""
    import numpy as np
    out = []
    for p in P:
        f = set(p["slots"].get("feats", "").lstrip("*").split("+"))
        ok = not (f & NORM_SUBSPACE_DENY) and not any(x.startswith(("hc_", "hyper")) for x in f)
        ok = ok and not p["dims"].get("emb_norm")
        if host:
            ok = ok and "tied_emb" not in f
        out.append(ok)
    return np.array(out, bool)


def emulates(B, A, mode="payable"):
    """B can emulate A under the HEADLINE relation (mode="payable", see emul_matrix): B dominates A
    on every capacity axis (with the TransMLA / MoE-depth / block / norm-mirror penalties), and every
    slot where they differ is crossed by an edge that has a currency -- exact, a floor / near-gap /
    regime, or a limit. A slot difference with NO currency (a RoPE host for a learned-abs target, a
    SiLU host for a situ target, a causal host for a diffusion target) blocks emulation.

    mode="capacity" is the secondary "if every type wall were free" relation (capacity + the one-way
    AR <= diffusion wall only); "exact" is the stricter slice and "mixgap" the looser one that grants
    the unpriced softmax <- SSM near-gap (MIXER_GAP_PAYABLE). Same matrix the frontier
    uses, evaluated on the pair."""
    return bool(emul_matrix([A, B], mode=mode)[0, 1])


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>model subset graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
 body{font-family:Helvetica,Arial,sans-serif;margin:0;background:#0f141a;color:#c8d2dd}
 #net{width:100vw;height:90vh;background:#0f141a;border-bottom:1px solid #2a333f}
 #bar{padding:8px 14px;font-size:13px;color:#c8d2dd;display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#131922}
 #bar .txt{flex:1;min-width:340px}
 b{color:#eef2f7}
 button{font:inherit;padding:4px 10px;border:1px solid #3a4655;border-radius:5px;background:#1c242f;color:#c8d2dd;cursor:pointer}
 button.on{background:#b8860b;color:#0f141a;border-color:#8a6508}
 .swatch{display:inline-block;width:10px;height:10px;border:2px solid #e0a92a;border-radius:50%;vertical-align:middle}
 div.vis-tooltip{background:#1c242f !important;color:#e8edf3 !important;border:1px solid #3a4655 !important;border-radius:5px;font-family:Helvetica,Arial,sans-serif;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,0.5)}
</style></head><body>
<div id="bar">
 <span class="txt"><b>Model subset graph.</b> Arrow A&rarr;B = "A is a special case of B" (B can be parameterized to emulate A).
 Node size &prop; how many models it emulates. <b>Click</b> a node to highlight everything it is a subset of; hover for its parent/child counts.</span>
 <input id="search" placeholder="&#128269; find a model&hellip;" oninput="doSearch(this.value)"
   style="font:inherit;padding:4px 10px;border:1px solid #3a4655;border-radius:5px;background:#1c242f;color:#e8edf3;min-width:180px">
 <span id="shits" style="color:#8a97a6;min-width:56px"></span>
 <button id="fbtn" onclick="toggleFrontier()"><span class="swatch"></span> Show only the frontier</button>
 <button id="lbtn" onclick="toggleLayers()">Sort by layers &rarr;</button>
</div>
<div id="net"></div>
<script>
const NODES = __NODES__;
const EDGES = __EDGES__;
NODES.forEach(n => { const d=document.createElement('div'); d.innerHTML=n.info; n.title=d; });
const nodes = new vis.DataSet(NODES);
const edges = new vis.DataSet(EDGES);
let frontierOn=false;
const net = new vis.Network(document.getElementById('net'), {nodes, edges}, __OPTS__);
net.once('stabilizationIterationsDone', () => net.setOptions({physics:false}));
// one batched DataSet write per action (per-node update loops lag on hundreds of nodes)
function batch(fn){ nodes.update(NODES.map(fn)); }
// outgoing adjacency (subset -> superset) for click-to-highlight-ancestors
const up={}; EDGES.forEach(e=>{(up[e.from]=up[e.from]||[]).push(e.to);});
net.on('click', p=>{
  if(frontierOn) return;                       // frontier view active: let the button drive it
  if(!p.nodes.length){ batch(n=>({id:n.id,opacity:1})); return; }
  const seen=new Set(), q=[p.nodes[0]];
  while(q.length){const x=q.pop(); (up[x]||[]).forEach(y=>{if(!seen.has(y)){seen.add(y);q.push(y);}});}
  batch(n=>({id:n.id, opacity:(n.id===p.nodes[0]||seen.has(n.id))?1:0.12}));
});
function toggleFrontier(){
  frontierOn=!frontierOn;
  const b=document.getElementById('fbtn');
  b.classList.toggle('on',frontierOn);
  b.lastChild.textContent = frontierOn ? ' Show all models' : ' Show only the frontier';
  batch(n=>({id:n.id, hidden: frontierOn && !n.frontier, opacity:1}));  // hidden = reliable everywhere
  net.fit();
}
let layersOn=false;
function toggleLayers(){                          // arrange nodes left->right by depth (eff_layers)
  layersOn=!layersOn;
  const b=document.getElementById('lbtn');
  b.classList.toggle('on',layersOn);
  if(layersOn){
    net.setOptions({physics:false});
    const byL={}; NODES.forEach(n=>{(byL[n.lyr]=byL[n.lyr]||[]).push(n.id);});
    const cols=Object.keys(byL).map(Number).sort((a,b)=>a-b);
    const upd=[];
    cols.forEach((L,ci)=>{const ids=byL[L], m=ids.length;
      ids.forEach((id,ri)=>upd.push({id, x: ci*210 - cols.length*105, y:(ri-(m-1)/2)*36, fixed:true}));});
    nodes.update(upd); net.fit();
    b.textContent='Free layout';
  } else {
    nodes.update(NODES.map(n=>({id:n.id, fixed:false})));
    net.setOptions({physics:true}); net.fit();
    b.innerHTML='Sort by layers &rarr;';
  }
}
function doSearch(q){
  q=q.trim().toLowerCase();
  document.getElementById('shits').textContent='';
  if(!q){ batch(n=>({id:n.id,opacity:1})); return; }
  const hits=NODES.filter(n=>((n.label||'')+' '+(n.info||'')).toLowerCase().includes(q)).map(n=>n.id);
  const hs=new Set(hits);
  batch(n=>({id:n.id, opacity: hs.has(n.id)?1:0.1}));           // dim non-matches, keep matches lit
  document.getElementById('shits').textContent = hits.length ? hits.length+' hit'+(hits.length>1?'s':'') : 'no match';
  if(hits.length) net.focus(hits[0], {scale:0.85, animation:{duration:300}});   // fly to the first match
}
if(!NODES.some(n=>n.frontier)) document.getElementById('fbtn').style.display='none';
</script></body></html>"""


COLOR = {"vanilla": "#dfe7f2", "glu": "#d7ead0", "moe": "#f6e2c0",
         "moe_shared": "#f2cdbb", "AffineNorm": "#ffe08a", "hadamard": "#e4dcef"}
HIER_OPTS = ("{layout:{hierarchical:{enabled:true,direction:'DU',sortMethod:'directed',"
             "levelSeparation:140,nodeSpacing:170}},physics:false,"
             "nodes:{shape:'box',scaling:{min:14,max:46},font:{size:13,multi:'html'},borderWidth:1,margin:8},"
             "edges:{arrows:'to',color:{color:'#9aa',highlight:'#c0392b'},font:{size:10},smooth:{type:'cubicBezier'}},"
             "interaction:{hover:true,tooltipDelay:80}}")
def _write_html(nodes, elist, opts, path):
    import json
    html = (HTML_TEMPLATE.replace("__OPTS__", opts).replace("__NODES__", json.dumps(nodes))
            .replace("__EDGES__", json.dumps(elist)))
    with open(path, "w") as f:
        f.write(html)


def export_html(classes, path):
    keys = list(classes)
    ids = {k: i for i, k in enumerate(keys)}
    edges = {}
    for a in keys:
        for b in keys:
            if a != b and le(classes[a]["slots"], classes[b]["slots"])[0]:
                edges[(a, b)] = le(classes[a]["slots"], classes[b]["slots"])[1]
    cover = {(a, b): d for (a, b), d in edges.items()
             if not any((a, c) in edges and (c, b) in edges for c in keys if c not in (a, b))}
    nodes = []
    for k in keys:
        s, mem = classes[k]["slots"], classes[k]["members"]
        n = len(mem)
        info = (f"<b>{slot_str(s)}</b><br>"
                + ", ".join(short(m) for m in mem[:15]) + (" ..." if n > 15 else ""))
        nodes.append(dict(id=ids[k], label=short(mem[0]) + (f"  +{n-1}" if n > 1 else ""),
                          value=n, info=info, color={"background": COLOR.get(s["ffn"], "#eee"), "border": "#33404f"}))
    elist = []
    for (a, b), diffs in cover.items():
        cur = ",".join(sorted({c for (_, cost) in diffs if cost for c in cost[0]}))
        exact = all(cost[1] for (_, cost) in diffs if cost)
        elist.append(dict(**{"from": ids[a], "to": ids[b]}, label=cur, dashes=not exact))
    _write_html(nodes, elist, HIER_OPTS, path)
    n_multi = sum(1 for k in keys if sum(1 for (a, _) in cover if a == k) > 1)
    print(f"wrote {path}: {len(nodes)} class nodes, {len(cover)} covering edges "
          f"({n_multi} are a subset of >1 parent)")


MODES = ("payable", "capacity", "exact", "nomix", "mixgap")


def _mode(mode, exact, capacity_only):
    """Resolve the relation: mode wins; the legacy flags map exact=True -> 'exact', 'nomix' ->
    'nomix', capacity_only=True -> 'capacity', default (exact=False / None) -> 'payable'."""
    if mode is not None:
        assert mode in MODES, mode
        return mode
    if capacity_only:
        return "capacity"
    return {True: "exact", "nomix": "nomix"}.get(exact, "payable")


CAT_SLOTS = ("norm", "pos", "act", "qknorm", "softcap", "sink", "mtp", "gen",    # + mixer / window / cla
             "head", "input")                                                 # (round 3, nonlm_heads)


def _subsets_hall(reach=None):
    """Every non-empty set X of target layer kinds with N(X) = the host kinds that reach some kind in X."""
    from itertools import combinations
    reach = reach or KIND_REACH
    out = []
    for r in range(1, len(LAYER_KINDS) + 1):
        for X in combinations(LAYER_KINDS, r):
            N = tuple(h for h in LAYER_KINDS if any(t in reach[h] for t in X))
            out.append((X, N))
    # keep only the binding ones: drop X if a superset X' has the same N(X') (its demand is >=)
    return [(X, N) for X, N in out if not any(set(X) < set(X2) and N2 == N for X2, N2 in out)]


_HALL = _subsets_hall()


def _kind_count(p, k, which):
    """A point's count of layer kind k: dims_of's k{k}_{which}; for points built without them (synthetic
    ones) derived from the ZERO_EDGES counts -- full = attention with a full window, local = the other
    attention layers, the rest recurrent (a non-softmax mixer) or KV-sharing (a CLA stack)."""
    d = p["dims"]
    key = f"k{k}_{which}"
    if key in d:
        return d[key]
    tot = d.get("eff_layers" if which == "eff" else "param_layers", 0)
    a, f = d.get(f"attn_{which}", tot), d.get(f"full_{which}", d.get(f"attn_{which}", tot))
    s = p["slots"]
    rest = max(tot - a, 0)
    return {"FU": f, "LU": max(a - f, 0), "R": rest if s.get("mixer", "softmax") != "softmax" else 0,
            "S": rest if (s.get("mixer", "softmax") == "softmax" and s.get("crosslayer_kv") == "yes") else 0
            }.get(k, 0)


def _kind_seq(toks, base, eff):
    """Round 3 (kind_order): the dims_of `kseq` string of a layer-kind token list (_attn_counts, in layer order) --
    runs "FU*3,R*1,..." over the UNROLLED depth eff (a loop repeats its listed block: eff a multiple of the listed
    base; any other ratio scales each run, as the counts are scaled); "" for a single plain kind (the counts decide
    there: an embedding of one kind is a count, Lean FrMixer assign_run)."""
    if not toks or (len(set(toks)) == 1 and "|" not in toks[0]):
        return ""
    runs = []
    for t in toks:
        if runs and runs[-1][0] == t:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    if eff != base and base > 0:
        if eff % base == 0:
            runs = [list(r) for _ in range(eff // base) for r in runs]
        else:
            runs = [[t, round(c * eff / base)] for t, c in runs]
    out = []
    for t, c in runs:
        if c <= 0:
            continue
        if out and out[-1][0] == t:
            out[-1][1] += c
        else:
            out.append([t, c])
    return ",".join(f"{t}*{c}" for t, c in out)


def _kseq_runs(s):
    return tuple((t, int(c)) for t, c in (x.split("*") for x in s.split(","))) if s else ()


def _tok_reach(reach):
    """Host token -> the target tokens it plays (kind_order): a plain kind X its KIND_REACH targets (and a parallel
    Y|R layer where X reaches both Y and R: the mixgap near-gap), a parallel X|R layer X's targets, R, and Y|R for
    every Y that X reaches (zero one mixer, or neither)."""
    att = ("FU", "FT", "LU", "LT", "S")
    out = {}
    for h in LAYER_KINDS:
        r = set(reach[h])
        out[h] = r | ({f"{y}|R" for y in reach[h] if y in att} if "R" in reach[h] else set())
        if h != "R":
            out[f"{h}|R"] = r | {"R"} | {f"{y}|R" for y in reach[h] if y in att}
    return out


def _seq_embeds(t, h, R):
    """Greedy earliest fit of the target runs t into the host runs h, in order, each target layer on its own host
    layer whose token reaches it (R: host token -> target tokens). Greedy is exact for an order-preserving
    embedding under a fixed pairwise predicate: by induction the greedy position of target layer i is <= its
    position in any embedding (the first reaching host layer after greedy(i - 1) <= the embedding's, which is after
    the embedding's i - 1 >= greedy(i - 1))."""
    j, left = 0, (h[0][1] if h else 0)
    for tok, c in t:
        need = c
        while need:
            if j >= len(h):
                return False
            if tok in R.get(h[j][0], ()):
                take = min(need, left)
                need -= take
                left -= take
            else:
                left = 0
            if left == 0:
                j += 1
                left = h[j][1] if j < len(h) else 0
    return True


_FULL_TOK = {"LU": "FU", "LT": "FT"}


def _seq_tf(runs, full):
    """A window-free side's runs with its sliding layers read as full ones (host_full / conv)."""
    if not full:
        return runs
    out = []
    for t, c in runs:
        b, _, r = t.partition("|")
        t2 = _FULL_TOK.get(b, b) + ("|" + r if r else "")
        if out and out[-1][0] == t2:
            out[-1] = (t2, out[-1][1] + c)
        else:
            out.append((t2, c))
    return tuple(out)


def _uniform_runs(p):
    """The runs of a point whose kseq is "" (one plain kind): that kind, its eff count (none: ())."""
    nz = [(k, int(_kind_count(p, k, "eff"))) for k in LAYER_KINDS]
    nz = [(k, c) for k, c in nz if c > 0]
    return tuple(nz) if len(nz) == 1 else ()


def kind_order_ok(t_seq, h_seq, factor=1, add=0, conv=False, hfull=False, reach=None):
    """Round 3 (kind_order): the target's layer-kind sequence (runs) embeds IN ORDER in the host's (each target layer
    to a later host layer whose kind reaches it; the unused host layers zeroed: o_proj = 0 and down_proj = 0, Lean
    zeroed_sublayer / FrMixer padded_run). factor k: each target layer takes k consecutive host layers of a reaching
    kind (the 2x depth); add: extra full-attention layers BEFORE the target's first (the start-token rule's marker
    blocks); conv / hfull: the window-free target / host side's sliding layers read as full ones."""
    R = _TOK_REACH_BASE if reach is None else (reach if "FU|R" in reach else _tok_reach(reach))
    t = _seq_tf(t_seq, conv)
    if factor != 1:
        t = tuple((x, c * factor) for x, c in t)
    if add:
        t = (("FU", int(add)),) + t
    return _seq_embeds(t, _seq_tf(h_seq, hfull), R)


_TOK_REACH_BASE = _tok_reach(KIND_REACH)
def _rot_points(P):
    """fr-positions (fp_nope_layer_host): P with each point's attention kind counts k{FU,FT,LU,LT}_{eff,param} replaced
    by its rotating ones (dims rk*), for kind_match; None when no point's differ (the check is kind_match itself)."""
    keys = [f"{k}_{w}" for k in ("FU", "FT", "LU", "LT") for w in ("eff", "param")] + ["seq"]
    if not any(p["dims"].get("rk" + x, p["dims"].get("k" + x)) != p["dims"].get("k" + x) for p in P for x in keys):
        return None
    # (release: kseq too -- the rotating layers' sequence rkseq, so kind_match's order check embeds in the RoPE layers)
    return [{"slots": p["slots"], "dims": {**p["dims"], **{"k" + x: p["dims"]["rk" + x] for x in keys
                                                           if "rk" + x in p["dims"]}}} for p in P]


def kind_match(P, which="eff", factor=1, reach=None, host_full=None, add_fu=None):
    """ok[i, j]: the layers of target P[i] can be assigned to distinct layers of host P[j], each to a host
    layer whose kind reaches it (KIND_REACH) -- Hall's condition, demand(X) <= supply(N(X)) for every set
    X of target kinds. which = 'eff' (unrolled) or 'param' (weight-distinct). factor (scalar or [i, j]
    matrix) multiplies the demand: the 2x depth of a dense host for a MoE target / a parallel host for a
    sequential one, charged per kind as the ZERO_EDGES charge it. reach: an override of KIND_REACH (the
    looser relations: mode="mixgap" lets attention play recurrent and shared-KV layers). Points without
    the kind dims (synthetic ones) demand and supply nothing, so the check is vacuous for them.
    host_full (bool per point, round 2 pass 5): the host's window is free config (WINDOW_FREE_TYPES), so its
    sliding layers supply as full ones (LU -> FU, LT -> FT). add_fu ([i, j] ints): extra full-attention layers
    the pair needs (the start-token rule's two blocks, a compensator layer), added to the FU demand."""
    import numpy as np
    reach = reach or KIND_REACH
    hall = _HALL if reach is KIND_REACH else _subsets_hall(reach)
    cnt = {k: np.array([_kind_count(p, k, which) for p in P], dtype=np.int64) for k in LAYER_KINDS}
    sup, conv = dict(cnt), None
    if host_full is not None and np.any(host_full):
        hf = np.asarray(host_full, bool)
        sup["FU"] = np.where(hf, cnt["FU"] + cnt["LU"], cnt["FU"])
        sup["FT"] = np.where(hf, cnt["FT"] + cnt["LT"], cnt["FT"])
        sup["LU"] = np.where(hf, 0, cnt["LU"])
        sup["LT"] = np.where(hf, 0, cnt["LT"])
        conv = hf[:, None] & ~hf[None, :]        # (transitivity: a window-free target on a fixed-window host
    #                                              demands its sliding layers as full ones)
    ok = np.ones((len(P), len(P)), bool)
    for X, N in hall:
        demand = sum(cnt[t] for t in X)
        supply = sum(sup[h] for h in N)
        dem = demand[:, None] * factor
        if conv is not None:
            dem = np.where(conv, sum(sup[t] for t in X)[:, None] * factor, dem)
        if add_fu is not None and "FU" in X:
            dem = dem + add_fu
        ok &= supply[None, :] >= dem
    if which == "eff" and _r3("kind_order"):
        ok &= _order_mask(P, ok, factor, reach, host_full, add_fu, conv)
    return ok


def _order_mask(P, ok, factor, reach, host_full, add_fu, conv):
    """Round 3 (kind_order): kind_match's count (Hall) is necessary, not sufficient, for an order-preserving
    embedding: target [attention, recurrent] on host [recurrent, attention] passes the count, and zeroing cannot
    reorder pre-norm residual layers (a zeroed layer is the identity, so the host computes its live layers in their
    own order). The pairs left to decide: a host whose kseq is "" (one plain kind) adds nothing to Hall (every
    target kind reached by that kind and the total within its depth, Hall's X = all kinds); a target whose kseq is
    "" and gets no extra marker layers neither (one kind: a count). The rest run kind_order_ok (memoised)."""
    import numpy as np
    seqs = [p["dims"].get("kseq", "") or "" for p in P]
    if not any(seqs):
        return np.ones_like(ok)
    n = len(P)
    hs = np.array([bool(x) for x in seqs])
    ts = hs.copy()
    fac = np.broadcast_to(np.asarray(factor), (n, n)) if np.ndim(factor) else None
    add = np.asarray(add_fu) if add_fu is not None else None
    cand = ok & hs[None, :] & (ts[:, None] | ((add > 0) if add is not None else False))
    out = np.ones_like(ok)
    hf = np.zeros(n, bool) if host_full is None else np.asarray(host_full, bool)
    runs = [_kseq_runs(x) if x else _uniform_runs(p) for x, p in zip(seqs, P)]
    cand &= np.array([bool(r) for r in runs])[:, None]      # (no single kind and no kseq: the counts decide)
    R = _TOK_REACH_BASE if reach is None else _tok_reach(reach)
    memo = {}
    for i, j in zip(*np.nonzero(cand)):
        key = (seqs[i] or runs[i], seqs[j], int(fac[i, j]) if fac is not None else int(factor),
               int(add[i, j]) if add is not None else 0, bool(conv[i, j]) if conv is not None else False,
               bool(hf[j]))
        v = memo.get(key)
        if v is None:
            v = memo[key] = kind_order_ok(runs[i], runs[j], key[2], key[3], key[4], key[5], R)
        out[i, j] = v
    return out


def _ffn_widths_fit(t, h):
    """Target FFN layers t into host FFN layers h, each target layer on its own host layer at least as wide: the
    sorted widths dominate (h has >= len(t) layers and its k-th widest >= t's k-th widest). Reflexive and
    transitive. A side is a tuple of widths, or (width, n_layers) for a model whose layers are all one width."""
    tu, hu = len(t) == 2 and not isinstance(t, tuple), len(h) == 2 and not isinstance(h, tuple)
    if tu and hu:
        return h[0] >= t[0] and h[1] >= t[1]
    if tu:
        return sum(1 for x in h if x >= t[0]) >= t[1]
    if hu:
        return h[0] >= max(t) and h[1] >= len(t)
    if len(h) < len(t):
        return False
    return all(a <= b for a, b in zip(sorted(t, reverse=True), sorted(h, reverse=True)))


# model types whose rotary embedding(s) take the rope type from the config through ROPE_INIT_FUNCTIONS (transformers
# 5 modeling_*.py, else 4.57 for types transformers 5 lacks, else the remote modeling file on the hub; scanned for the
# zoo's RoPE types, round 2 pass 2), so rope_type "longrope" with per-pair short_factor / long_factor lists sets any
# frequency per pair. Not: remote families with their own rotary (InternLM / InternLM2, Baichuan, Qwen-1, MiniCPM,
# OpenELM, ChatGLM, Arctic, DeepSeek-MoE, ...), GPT-J / CodeGen (a fixed sinusoid table), DBRX, code not on the hub.
ROPE_PAIRS_FREE_TYPES = frozenset({
    "afmoe", "apertus", "arcee", "axk1", "axk2", "bailing_hybrid", "bamba", "beam", "cohere", "cohere2",
    "deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4", "diffusion_gemma_text", "doge", "dots1", "ernie4_5",
    "ernie4_5_moe", "exaone4", "exaone_moe", "falcon", "falcon_h1", "flex_olmo", "flint", "gemma", "gemma2",
    "gemma3", "gemma3_text", "gemma4", "gemma4_text", "gemma4_unified_text", "glm", "glm4", "glm4_moe",
    "glm4_moe_lite", "glm_moe_dsa", "gpt_neox", "gpt_neox_japanese", "gpt_oss", "granite", "granite_swa",
    "granitemoe", "granitemoe_swa", "granitemoehybrid", "helium", "hrm_text", "hunyuan_v1_dense", "hunyuan_v1_moe",
    "hy_v3", "hy_v4", "hyperclovax", "jetmoe", "kolibri1", "laguna", "lfm2", "lfm2_moe", "llama", "llama4",
    "llama4_text", "mellum", "mimo_v2_flash", "minicpm3", "minimax", "minimax_m2", "minimax_m3_vl", "ministral",
    "ministral3", "mistral", "mistral4", "mistral_large4_text", "mixtral", "moshi", "muse_glimmer",
    "muse_glimmer_text", "muse_spark_text", "nemotron", "olmo", "olmo2", "olmo3", "olmoe", "persimmon", "phi",
    "phi3", "phimoe", "qwen2", "qwen2_5_vl", "qwen2_moe", "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen3_5_text", "qwen3_moe", "qwen3_next", "qwen3_vl", "qwen3_vl_moe", "qwen3_vl_moe_text", "qwen3_vl_text",
    "qwen4_exp", "qwen4_exp_text", "seed_oss", "smollm3", "solar_open", "stablelm", "starcoder2", "vaultgemma",
    "youtu", "zamba2",
    # remote modeling files that build their rotary through ROPE_INIT_FUNCTIONS[rope_type] (hub code, round 2 pass 2)
    "bailing_moe", "k2_horizon", "mimo_v2", "nemotron-nas", "llada2_moe", "exaone", "ouro", "sdar", "sdar_moe",
    "mask_moe", "step3p5", "dream", "xing4_0", "bolmo", "joyai_llm_flash", "internlm3", "step3_text", "param2moe",
    "ming_v3_asr", "sarvam_moe", "apriel", "youtu_vita_text", "youtu_vl", "motif", "kanana2_tiny", "jet_nemotron",
    "iquestloopcoder", "interns1_pro_text",
    # remote modeling files that run a native transformers LM body (its rotary): MiMo (Qwen2Attention), FastVLM /
    # VideoLLaMA3 / Hulu-Med (Qwen2 / Qwen3 models), the bge Gemma-2 reranker (Gemma2RotaryEmbedding), Typhoon2-audio
    # (Llama), Naive-N0.5 (GPTNeoXRotaryEmbedding), gpt-oss-Puzzle, qwen3loop, Isaac (Qwen3), gemma3moe, Magma
    # (AutoModelForCausalLM on a Llama-3 text config), AllGDN / Qwen3.8-Longbook (Qwen3.5), simulvoice (Qwen2)
    "mimo", "llava_qwen2", "videollama3_qwen2", "hulumed_qwen2", "hulumed_qwen3", "cost_wise_gemma",
    "typhoon2audio", "naive_n05_flash", "gpt_oss_puzzle", "qwen3loop", "isaac", "gemma3_moe", "magma", "allgdn",
    "qwen3_5_gdn24", "simulvoice_runtime_probe"})


# Round 3 (nope_rope_limit): NoPE on a RoPE host as a LIMIT -- a HOST CONFIG LEVER (a design ruling: rotation rate
# -> 0 is a valid edge). rope_theta alone cannot stop the first pair (Lean rope_theta_cannot_reach_nope: theta^0 = 1),
# but LINEAR rope scaling divides every position by its factor s (transformers 4.57.6 modeling_rope_utils.py:166-175,
# _compute_linear_scaling_rope_parameters: inv_freq /= factor; ROPE_INIT_FUNCTIONS["linear"]): every angle (i / s) w_p
# -> 0 as s -> inf and the logit tends to the unrotated q . k uniformly on bounded positions (proofs/lean/
# EmulationProofs/Position.lean rope_scale_limit_identity / rope_scale_limit_bound / rope_scale_limit_attn; PROOFS.md
# "NoPE <= RoPE, linear rope scaling to infinity"). A limit, never exact at a finite s: payable / mixgap only.
# Measured on a real NoPE target (attn/nope_rope_limit_agreement.py, attn/data/nope_rope_limit_agreement.jsonl):
# granite-4.0-h-350m-base on its own family switched to RoPE, linear factor 1e4 .. 1e12: top-1 100%, 32-token
# continuations 100% identical (factor 1, plain RoPE: 71.8% / 0%).
# Which families read linear scaling (the code, per family):
#   - installed transformers (4.57.6) and transformers main: every rotary that honours "linear" does so
#     through ROPE_INIT_FUNCTIONS[rope_type] (the `rope_type != "default"` branch), which also offers "longrope" -- the
#     ROPE_PAIRS_FREE_TYPES families above, whose zero frequency is already exact (zero_freq_room); the families with
#     their own rotary read no scaling (GPT-J, CodeGen: fixed sin / cos tables) or refuse it (RecurrentGemma raises on
#     any rope type but "default", modeling_recurrent_gemma.py:69-74): walls for this route.
#   - outside the pairs-free table: dots3_note (tf-main modeling_dots3_note.py:82-101) and LFM3 (tf-main
#     modeling_lfm3.py:72-82; the remote modeling_lfm3*.py compute_rope_parameters, ROPE_INIT_FUNCTIONS on the full
#     head, then the zero tail) build ONE TABLE PER LAYER TYPE, each from its own rope_parameters[layer_type]
#     (ROPE_INIT_FUNCTIONS, linear included) -- per-table levers (ROPE_LINEAR_PER_TABLE); Orion (modeling_orion.py:
#     264-281), MiniCPM (modeling_minicpm.py:957-978 / :856-877: linear, dynamic, longrope) and Bunny-Phi
#     (modeling_bunny_phi.py:1105-1122) branch on config.rope_scaling["type" / "rope_type"] == "linear" into a
#     LinearScalingRotaryEmbedding: one table, scaled (remote code read as text on disk, never run -- the pairs-free
#     table's hub-code precedent).
# rope_lin (dims_of) = 2 for a per-table family, 1 for a one-table one (exactly one rotating table: the factor then
# reaches every rotating layer; a multi-table host whose tables a single rope_scaling may not all reach -- Gemma-3's
# local table is pinned to "default" -- gets 0, conservatively: those are pairs-free anyway), else 0.
ROPE_LINEAR_ONE_TABLE = _R3Set(add={"nope_rope_limit": ROPE_PAIRS_FREE_TYPES | {"orion", "minicpm", "bunny-phi"}})
ROPE_LINEAR_PER_TABLE = _R3Set(add={"nope_rope_limit": {"dots3_note", "lfm3", "lfm3_rd", "lfm3_omni"}})


def _rope_lin(d, sched):
    """rope_lin of a config (round 3, nope_rope_limit): 2 = one linear factor per table (per-layer-type rope
    parameters), 1 = one factor for the model's one rotating table, 0 = the code reads no linear scaling (or a
    multi-table host one factor may not reach entirely)."""
    if not _r3("nope_rope_limit"):
        return 0
    mt = str(d.get("model_type") or "").lower()
    rot = [n for n, _ in sched if n > 0]
    if mt in ROPE_LINEAR_PER_TABLE and rot:
        return 2
    return 1 if mt in ROPE_LINEAR_ONE_TABLE and len(rot) == 1 else 0


def nope_rope_limit_slot(v_t, rope_lin_h, rot_h):
    """The NoPE-target half of the nope_rope_limit lever (a host config lever, a LIMIT): a NoPE target on a (partial)
    RoPE host whose code reads linear scaling for its every rotating table (rope_lin > 0) -- factor -> inf, every
    angle -> 0 (Lean rope_scale_limit_identity). Elementwise over (target pos value, host rope_lin, host is RoPE)."""
    return bool(_r3("nope_rope_limit") and v_t == "nope" and rot_h and rope_lin_h > 0)


def nope_rope_limit_rot(need_t, hd_t, kind_t, pf_t, ff_t, cap_h, hd_h, pf_h, ff_h, kind_h, rope_lin_h):
    """The mixed-target half of the nope_rope_limit lever: a RoPE target with NoPE layers (negative rope_sched entries,
    zero-frequency slots) on a host with ONE LINEAR FACTOR PER TABLE (rope_lin 2) and >= 2 rotating tables: one host
    table is scaled to infinity and read by the NoPE layers (every dim of their heads -> the identity rotation), the
    target's rotating schedules fit the host's OTHER tables by the host's own rule (rope_fits on a pairs-free host,
    fixed_rot_fits on a fixed one, the class conditions of the rot gate: a fixed host plays no pairs-free target, a
    fraction-free target only on a fraction-free host). A one-table host cannot: scaling its table to infinity also
    stops the target's rotating layers (they keep needing the exact zero-frequency slots, zero_freq_room / rope_fits).
    Layer alignment (which host layers read which table) is not modelled, as in rope_fits."""
    if not _r3("nope_rope_limit") or rope_lin_h < 2:
        return False
    pos = tuple(r for r in need_t if r > 0)
    if not pos or not any(r < 0 for r in need_t):
        return False
    if not pf_h and (pf_t or (ff_t and not ff_h)):
        return False
    for b, c in enumerate(cap_h):
        if c <= 0:
            continue
        rest = tuple(x for a, x in enumerate(cap_h) if a != b)
        if not any(x > 0 for x in rest):
            continue
        if pf_h:
            ok = rope_fits(pos, rest)
        else:
            ok = fixed_rot_fits(pos, hd_t, kind_t, rest, hd_h, ff_h, kind_h)
        if ok:
            return True
    return False


# ---------------------------------------------------------------------------------------------------------------
# Round 2, pass 5: the config rules, the rescues and the refutations.
#
# CONFIG RULES (decided). A HOST may set a shape-preserving config value its family's CODE reads (verify/levers/
# family_code.py: the config read in the transformers modeling file, out/family_code.json; remote code is not
# checked and gets nothing):
#   free      positions & norms (RoPE theta / scaling, partial_rotary_factor, norm eps, window sizes), the routing
#             numbers (n_group / topk_group, norm_topk_prob, routed_scaling_factor, router scoring: pass 3), scales &
#             caps (query_pre_attn_scalar, softcap values, embedding / residual / logit multipliers), norm on / off flags;
#   not free  the activation function, top-k, other feature switches (use_sliding_window itself, DSA index_topk);
#             shape-changing fields (tie_word_embeddings, attention_bias, mlp_bias, adding a qk-norm) never.
# A target is the function it computes: its config is never changed.
#
# WINDOW LEVER (verify/levers/out/window_audit.md, branch frontier-levers): in these families the sliding window is
# only the mask's parameter (and in Cohere-2 / EXAONE-4 / Kolibri the RoPE-on switch, which an int window keeps on),
# so a host sets it >= the context and every sliding layer is a full layer (kinds LU -> FU, LT -> FT; the ZERO_EDGES
# "only the host's full layers count for a full target" no longer binds for it). DeepSeek-V4-style compressed layers
# (deepseek_v4 / deepseek_v41_text / ernie5_5_moe) are excluded: with a full window they still attend to the
# compressor's pooled entries (kv_compress walls them anyway). The window_cfg families of family_code.json.
# Round 3, batch 2c: Llama-4's chunk size is config too (create_chunked_causal_mask reads config.attention_chunk_size),
# but its chunked layers are already counted full (FU: chunked_attention_unread), so no entry is needed. The remote
# families whose window keys sliding_window_keys_unread reads get NO entry (remote code: no lever); cost_wise_gemma
# (BAAI's reranker on transformers-4.42 Gemma2DecoderLayers) is NOT window-free -- a conservative design choice, and
# live now that it stays placed (nonlm_kept). Its code, read on disk (gemma_model.py): it reads no window key and its
# _update_causal_mask (:440-482) is the plain causal mask, so the band on the even layers comes from the imported
# Gemma2DecoderLayer (:56, :258) at config.sliding_window (4096 < max_position_embeddings 8192): the code USES the
# window. That import names 4.42-only classes (GEMMA2_ATTENTION_CLASSES, Gemma2FlashAttention2: the code runs only on
# the config's transformers 4.42.2), and the 4.42 layer that would read sliding_window as a host lever is not on disk
# here (installed: 4.57), so no lever is credited: not window-free (flagged; the spec's "optional, decided yes" would
# add it once the 4.42 Gemma2DecoderLayer is read). The table stays one config-class predicate (host_full / tgt_full),
# so the lever stays transitive:
WINDOW_FREE_TYPES = frozenset({
    "afmoe", "beam", "cohere2", "diffusion_gemma_text", "doge", "dots1", "dots3_note", "exaone4", "exaone_moe", "flint",
    "gemma2", "gemma3_text", "gemma3n_text", "gemma4", "gemma4_text", "gemma4_unified_text", "gpt_oss", "granite_swa",
    "granitemoe_swa", "kolibri1", "laguna", "lfm3", "mellum", "mimo_v2", "mimo_v2_flash", "minimax", "ministral",
    "ministral3", "mistral", "mixtral", "moshi", "muse_glimmer", "muse_glimmer_text", "muse_spark_text", "olmo3", "phi3",
    "phimoe", "qwen2", "qwen2_5_vl", "qwen2_moe", "qwen3", "qwen3_moe", "recurrent_gemma", "smollm3", "starcoder2",
    "vaultgemma"})
# PARTIAL-ROTARY LEVER: the family's attention splits q / k into rot / pass parts by partial_rotary_factor
# (family_code.json partial_rope: q_rot / q_pass / rotary_ndims in the modeling file), so a host may RAISE (or lower)
# its rotary fraction: its table then drives any number of rotated dims up to its head_dim. Not MLA, whose rotary
# width qk_rope_head_dim is a shape (the MLA families below are skipped by dims_of).
ROT_FRAC_FREE_TYPES = frozenset({
    "bailing_hybrid", "bamba", "codegen", "dots3_note", "glm", "glm4", "glm4_moe", "gpt_neox", "gpt_neox_japanese",
    "gptj", "hy_v4", "laguna", "mimo_v2", "mimo_v2_flash", "minimax_m2", "minimax_m3_vl", "nemotron", "persimmon",
    "phi", "phi3", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5_text", "qwen3_next", "recurrent_gemma",
    "stablelm"})
# round 3 (mm200_core_keys_unread): MM200's rotary width is the config's partial_rope_frac (modeling_mm200.py:125-126,
# rot_dim over a buffer inv_freq, q / k split at rot_dim :141-142): a shape-preserving position value (Lean
# partial_rotary_raise)
ROT_FRAC_FREE_TYPES = _R3Set(ROT_FRAC_FREE_TYPES, add={"mm200_core_keys_unread": {"mm200"}})# SOFTCAP VALUES: attn_logit_softcapping / final_logit_softcapping read from the config (family_code.json softcap_cfg);
# a null cap is no cap in the code, so such a host drops its caps EXACTLY (REGIME_PAYABLE already pays the large-c
# setting for every capped host; this adds the exact slice).
SOFTCAP_FREE_TYPES = frozenset({
    "beam", "diffusion_gemma_text", "gemma2", "gemma3_text", "gemma3n_text", "gemma4", "gemma4_text",
    "gemma4_unified_text", "muse_glimmer", "muse_glimmer_text", "muse_spark_text", "vaultgemma"})
# NORM ON / OFF FLAGS: a q / k norm built only when a config flag says so (family_code.json qknorm_switch): the host
# switches it OFF exactly (switching one ON adds weights: never free).
QKNORM_SWITCH_TYPES = frozenset({"cohere", "glm4_moe", "llama4_text", "persimmon", "phi", "stablelm"})
# (query_pre_attn_scalar, the embedding / residual / logit multipliers: no rule of the relation compares them, so
# freeing them changes no pair.)
#
# THE ROT GATE UNDER CONFIG-ONLY RULES (A3). The partial-RoPE host construction (hf/partial_rope_host.py) writes the
# target's frequencies into the host's inv_freq buffer. A config can do that only where the rotary code takes its rope
# type from the config with per-pair factors (longrope: ROPE_PAIRS_FREE_TYPES, "pairs-free"): there any frequency per
# pair, 0 included (a huge factor), is a config value. A FIXED-rotary host (its own rotary code, no rope_scaling read)
# can set only rope_theta (and, in ROT_FRAC_FREE_TYPES, its rotary fraction): Lean defaultFreq_match /
# partial_rope_host_config -- theta_h = theta_t^(c / d_t) puts a default-schedule target's d_t rotated dims on the
# host's first d_t / 2 pairs, the rest of the host's c rotated dims then hold zeros -- and rope_theta_cannot_reach_nope:
# inv_freq[0] = 1 for any theta, so a fixed full-RoPE host has NO zero-frequency slot. So a fixed host plays a RoPE
# target only when the target has ONE default-type schedule (rope_default) and its non-rotating dims fit the host's
# pass dims (hd_t - d_t <= hd_h - c; a fraction-free host sets c = d_t), and a target needing zero-frequency slots
# (NoPE / ALiBi / a learned-abs target under the start-token rule / a sink's spare pair) only within its pass dims.
#
# THE EPS LEVER (branch measure-batch2: graph/eps_lever.py, hf/data/eps_lever_agreement.jsonl; Lean EpsLever.lean).
# A host switches OFF a norm the target lacks: the weights writing the norm's input scale it by 2^-a, so the norm is
# linear (fl(ms + e) = e) and its gain 2^a sqrt(e) makes it the identity. Measured 100% / 100% (fp32 and fp64, at the
# rounding floor) for qk-norm, sandwich post-norms, Gemma-4's v_norm and the MLA kv_a_layernorm. Exact up to rounding:
# payable, labelled a LIMIT (not in the exact slice). Not for an eps = 0 norm, a post-only block (OLMo-2: the post-norm
# is its only norm): dims eps_lever. The fp16 exclusion (the scaled tensors leave fp16's range) is a RUN-TIME caveat,
# not a property of the stored checkpoint: the lever holds for a host RUN in bf16 / fp32 (the emulator is run there,
# whatever dtype the checkpoint was stored in), so no host is walled for an fp16 torch_dtype (round 3, a design
# decision: the run-dtype reading of pass 5, kept; dims_of below). A gain is needed to take
# 2^a sqrt(e) (a normalised vector has rms <= 1): the weight-free variants (head_free, dsv4's q) and the LayerNorm ones
# (the centring is not the identity on a linear map's output) are not switched off by it.
EPS_LEVER_QKNORM = frozenset({"head", "head_distinct", "head_post", "proj", "qkv_input"})
# round 3, batch 2a: V4.1's weighted K = V norm (its only norm: the lever's k = v flag, Lean leverAttn_layer_bound) and
# dots3-note's weighted rope-key norm (the k flag; RoPE after the norm is inside B(q, k)) -- both gained RMS norms with
# eps = rms_norm_eps > 0. (head_qfree / head_free_gain / ngpt keep a weight-free norm the lever cannot switch off.)
EPS_LEVER_QKNORM = _R3Set(EPS_LEVER_QKNORM, add={"dsv41_q_head_norm_absent": {"dsv4_kv"},
                                                 "mla_k_rope_norm": {"mla_krope"}})
EPS_LEVER_FEATS = frozenset({"post_norm", "vnorm"})
# The same lever reads per-norm effective eps (a norm's input scaled by s has eps_eff = eps / s^2): it relaxes the two
# pass-3 / pass-4 "integer multiple" rules -- the MLA latents (eps hard-coded 1e-6: a zero-padded wider latent's
# rms is r = kl_t / kl_h of the target's, scale the latent input by 1 / sqrt(r)) and Mamba-2's grouped gated norm (the
# config eps shared with the stream norms: eps_h = eps_t w_t / w_h, the stream rescaled for the others; not Zamba2,
# whose gated-norm eps is 1e-5 in code) -- to host >= target on a lever-capable host (payable; the exact slice keeps
# the multiples).
#
# THE START-TOKEN RULES (branch bos-token: graph/bos_rules.py, hf/bos_constructions.py; Lean learned_abs_two_block,
# sink_sub_plain). The input begins with one fixed marker seen by the EMULATOR only: the target sees plain text, its
# position table shifted by one (T + 1 = the target's context + the marker).
#   learned-abs target on a position-free / RoPE / partial-RoPE host (pos field switched off: inv_freq = 0, which the
#     zero-frequency rule above prices): +2 layers (full window), +5 hidden (+10 on a LayerNorm host: mirrored), and
#     the two extra MLPs' units (shared variant: T + 1, and T + 1 + 5 erasers on a gated host / + 10 on a vanilla one);
#     host act with a ramp / step limit, an exact linear eraser on a vanilla host. Exact.
#   sink target on a sink-free host with a q bias (a k bias alone cancels in the softmax) and head_dim >= target + 2
#     (rotating: a zero-frequency pair) or + 1 (position-free), +1 hidden. Exact; up to the host's sliding window in a
#     windowed layer (the marker leaves it: the context caveat, as DSA's). A target that reads its own BOS needs the
#     marker before it (one more position, not priced: USE_CONTEXT off).
BOS_EXTRA_LAYERS = 2
BOS_EXTRA_HIDDEN = {"RMSNorm": 5, "LayerNorm": 10}
BOS_KNOT_ACTS = frozenset({"ReLU", "SiLU", "GeLU", "GeLU_tanh", "mish", "swigluoai", "relu2", "xielu", "situ"})
BOS_ODD_LINEAR_ACTS = frozenset({"ReLU", "SiLU", "GeLU", "GeLU_tanh", "swigluoai"})
BOS_ERASERS = {True: 5, False: 10}                 # gated host / vanilla host (two units per channel)
SINK_EXTRA_HIDDEN = 1
SINK_SPARE_ROT, SINK_SPARE_FLAT = 2, 1
# q bias from the config: explicit keys first, then the families whose q projection always has one (bos_rules)
_QBIAS_KEYS = ("attention_bias", "qkv_bias", "use_qkv_bias", "add_qkv_bias", "enable_bias", "attn_bias",
               "use_bias", "bias")
QBIAS_TYPES = frozenset({"qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl", "qwen", "gpt2", "gpt_bigcode",
                         "opt", "bloom", "gpt_neox", "phi", "starcoder2", "xglm", "gpt_oss",
                         "gpt_oss_puzzle", "glm", "chatglm"})
# round 3 (bias_qkv_by_code_unlisted): the Qwen2Attention bodies' q bias (BIAS_QKV_TYPES' round-3 entries)
QBIAS_TYPES = _R3Set(QBIAS_TYPES, add={"bias_qkv_by_code_unlisted": {
    "arkasr", "cambrian_qwen", "hulumed_qwen2", "llava-qwen2", "llava_qwen2", "meditor", "spatiallm_qwen",
    "step_audio_2", "sundial", "videollama3_qwen2"}})
# (round 3, batch 4b, cfrd_film_norm_shift: CFRD is NOT a q-bias host -- its constant into q is W_q 0.1 s_d, one shift
# shared by q / k / v / MLP, no independent q bias for the start-token / sink constructions; q_bias is read on hosts only)
#
# ACTIVATION WIDTH (branch act-width-fits: graph/act_width.py, act/data/act_width_fits.json, act_width_agreement.jsonl).
# A swap the one-unit floor table does not pay is payable at FFN width x k(t, h): k host units per target unit,
# sum_i alpha_i X(beta_i a + gamma_i), the shifts gamma_i only with a host gate bias (graph/act_width K_SHIFT); a
# swigluoai target's (up + 1) needs a host up bias whatever the host activation (Lean swigluoai_sub_silu: the graph's
# swigluoai <- SiLU cell was unconditional, verify/BRIDGES.md MISMATCH 3).
# GeLU erf <-> tanh (branch measure-batch2: graph/gelu_variants.py, act/data/gelu_swap_agreement.jsonl): payable both
# ways (six models, raw swap), approximate: the 0.0003 cells in ACT_FLOORS.
#
# LOOPED HOSTS (branch moe-loop-rescue: graph/rescue_rules.py, hf/looped_host_prenorm.py). An untied target on a looped
# host is payable only on a host whose every iteration re-runs fresh causal pre-norm self-attention over the current
# states without a qk-norm (only Nanbeige4.2 in the zoo): +2 channels (counter, start flag) and 2d + 2 FFN units (+2
# per layer if the host norms between loops, +4d per layer if it has a v bias), under the start-token rule. Every other
# looped host is a wall for an untied target (IQuest-Loop's loop-1 KV + window, ARLoop / kotodama attention
# residuals, Dory nGPT, LFM3-RD gated convs, qwen3loop / ARLoop-baseline qk-norm).
LOOP_PLAIN_TYPES = frozenset({"nanbeige"})
LOOP_NORM_BETWEEN_TYPES = frozenset({"nanbeige"})
LOOP_EXTRA_CHANNELS = 2
#
# PRE-NORM RESCUES (proofs/PROOFS.md "Pre-norm rescue attempts"; verify/prenorm_rescue/results/). The
# LayerNorm fold without an lm_head bias and parallel <= seq are refuted as exact but payable with a pad (host eps = 0,
# a config value) and a 1-D compensator of the token RMS: measured on Pythia-160m, GLU k = 8 100% (LN fold KL 3.6e-8,
# parallel 4.6e-8), ReLU + bias k = 16 99.97% / 99.76%. Payable only for target families whose measured RMS dynamic
# range D (r_max / r_min on the worst layer, position 0 excluded) is within what was tested, D <= 31, at k units per
# carried channel: the LN fold 1 channel (zeta) in one extra layer, +2 hidden (pad, zeta); parallel <= seq the d
# carried channels, k d units per layer, +1 hidden (the pad) on the 2x carry. Other families stay walls. Sinks without a
# q bias, the CLA carry, ALiBi on a learned-abs host (needs a channel proportional to j R_j: POS_ORDER edge removed) and
# post-norm targets stay walls (not measured end to end).
PRENORM_RMS_RANGE = {"gpt_neox": 30.8, "olmoe": 25.9, "llama": 16.2, "qwen3": 3.6, "gpt_oss": 3.7, "olmo2": 53.5}
PRENORM_D_MAX = 31.0
COMPENSATOR_K = {"glu": 8, "relu": 16}             # units per carried channel: a gated host, an ungated biased one
# lm_head with a bias (the LayerNorm fold's beta term: Lean layerNorm_sub_rmsNorm condition 8)
LM_HEAD_BIAS_TYPES = frozenset({"phi", "bunny-phi", "ctrl"})
# LayerNorms without a beta (the fold needs no constant readout): Cohere's LayerNorm(bias=False), OLMo-1's
# non-parametric LayerNorm
LN_NO_BETA_TYPES = frozenset({"cohere", "cohere2", "olmo", "hf_olmo"})
# Recurrent families whose output norm (Mamba-2's grouped gated RMSNorm, GDN / KDA's per-head gated norm) takes its
# eps from the config (family_code.json norm_eps_cfg; transformers Mamba2 layer_norm_epsilon), so the eps lever
# relaxes their "integer multiple" / "equal key head" rules (_rc_pair_ok_lever). Not Zamba2 (1e-5 in code), nor the
# unchecked remote families (Nemotron-H-Puzzle, CENO, ...).
REC_EPS_CFG_TYPES = frozenset({"mamba2", "nemotron_h", "bamba", "granitemoehybrid", "falcon_h1", "olmo_hybrid",
                               "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text",
                               "kimi_linear", "bailing_hybrid", "youtu", "glm5_next_text"})
# ---------------------------------------------------------------------------------------------------------------


# Leave-one-out switches for the pass-5 ports (graph/frontier's effects table; empty = every port on): a name here
# turns that port off -- a rescue / lever stops crossing its wall, a new wall reverts to the pass-4 rule.
P5_PORTS = ("window", "partial", "softcap", "qk_switch", "rot_config", "eps_qknorm", "eps_feats", "eps_mla_plain",
            "eps_latent", "eps_rec", "bos_labs", "bos_sink", "act_width", "gelu", "oai_bias", "loop", "comp_ln",
            "comp_par", "ln_beta", "alibi_labs",
            # fr-resnorm (dist/fr_resnorm.md): hyper-connection hosts switched off, DiffusionGemma's embedding norm
            # switched off, the LayerNorm-beta fold by a dominated final readout (each also gated by its round-3 key)
            "hc_off", "sc_norm_off", "ln_beta_readout")
P5_OFF = set()


def _p5(port):
    return port not in P5_OFF


# round 3, batch 3b: pass-5 ports whose credit a round-3 key removes (the code path kept, off while the key is on):
# port -> key. The reduction catalog's types read it (verify/spot/reduction_types.py: a removed port is not a headline
# type)
P5_REMOVED_BY = {"comp_ln": "spot_comp_removed", "comp_par": "spot_comp_removed"}


def p5_port_live(port):
    """A pass-5 port credits edges: on (not in P5_OFF) and not removed by a round-3 key (P5_REMOVED_BY)."""
    return _p5(port) and not (port in P5_REMOVED_BY and _r3(P5_REMOVED_BY[port]))


def act_width_k(target, host, gate_bias):
    """The activation-width multiple k of graph.act_width.act_k (0 = no fit), with the round-3 wall: under
    spot_act_width_oai (verify/spot/SPOT.md, catalog p5_act_width_gelu_oai) a swigluoai host branch offers no k-unit fit
    -- act/act_width_fit.py fits z sigmoid(1.702 z) and ignores gpt-oss's gate clamp min(g, 7) (swiglu_limit): the k = 6
    GeLU fit's betas reach 3.8, the clamp binds from |a| ~ 1.8, and with it the fit's error on [-6, 6] is 0.54, not
    4.8e-4 (top-1 97.5%, 2 / 8 identical). Not refitted under the clamp: a wall."""
    from graph.act_width import act_k
    if _r3("spot_act_width_oai"):
        hb = [b for b in branches(host) if b != "swigluoai"]
        if not hb:
            return 0
        host = "|".join(hb)
    k = act_k(target, host, bool(gate_bias), one_unit=_p5("gelu")) or 0
    if _r3("cost_act_k_measured") and k:
        # branch fr-cost: the cells the table over-charges against the MEASURED standard (ACT_K_MEASURED): a target
        # whose every branch has a measured cheaper k on some host branch, the k-table's own value elsewhere
        km = 1
        for t in branches(target):
            best = None
            for h in branches(host):
                kt = act_k(t, h, bool(gate_bias), one_unit=_p5("gelu"))
                if kt and not gate_bias:
                    kt = min(kt, ACT_K_MEASURED.get((ACT_K_ALIAS.get(t, t), h), kt))
                if kt and (best is None or kt < best):
                    best = kt
            km = max(km, best or k)
        k = min(k, km)
    return k


# branch fr-cost (cost_act_k_measured): a no-shift k-unit swap the K table charges one unit more than the measured
# standard needs. GeLU (erf and tanh) <- SiLU at k = 5: the stored fits miss ACT_FLOOR_MAX by 1% (erf 5.07e-4) and 11%
# (tanh 5.57e-4), and on every model measured (graph/act_width.AGREEMENT, act/data/act_width_agreement.jsonl, fp32,
# 24 prompts) the k = 5 swap is payable: Pythia-160m 100.0% / 95.8%, Pythia-410m 100% / 100%, Pythia-1.4b 100% / 100%,
# gpt2 99.87% / 91.7%, gpt2-medium 99.93% / 95.8%, Gemma-2-2b 99.87% / 95.8% (top-1 / identical 32-token). k = 4 fails
# on gpt2-medium (83.3%), so 5 is the measured k. The width calibration's own boundary (graph/act_width: "fits at 1e-3
# fail on gpt2-medium, 5.6e-4 passes") puts both fits on the payable side.
ACT_K_MEASURED = {("GeLU", "SiLU"): 5}
ACT_K_ALIAS = {"GeLU_tanh": "GeLU"}


P5_DIMS = ("win_free", "rot_frac_free", "rope_kind", "softcap_free", "qk_switch", "eps_lever", "q_bias", "ffn_bias",
           "lm_head_bias", "ln_beta", "loop_plain", "loop_norm", "loop_vbias", "rms_ok", "pos_table", "rec_eps_cfg",
           "sc_norm_off")                                     # (fr-resnorm)


def host_rope_cap(p):
    """A host point's rotary tables (rope_cap) after its config levers: a fraction-free host (rot_frac_free:
    partial_rotary_factor read by a rot / pass split) can raise any table to its whole head."""
    d = p["dims"]
    cap = tuple(d.get("rope_cap", d.get("rope_sched", ())))
    if d.get("rot_frac_free") and _p5("partial"):
        hd = int(d.get("head_dim", 0))
        cap = tuple(max(c, hd) if c > 0 else c for c in cap)
    return cap


def fixed_rot_fits(need, hd_t, kind_t, cap, hd_h, frac_free, kind_h="", own=None):
    """The rot gate on a FIXED-rotary host (not rope_pairs_free): only rope_theta (and, fraction-free, the rotary
    fraction) is config, so the target must be one schedule of d_t rotated dims of a type the host's code computes
    (kind_t "" = the default theta^(-2i/d), or the host's own kind_h), placed on the host's first d_t / 2 pairs
    (theta_h = theta_t^(c / d_t)), its hd_t - d_t non-rotating dims in the host's pass dims (hd_h - c): Lean
    defaultFreq_match, partial_rope_host_config. A fraction-free host sets c = d_t. A target with several tables
    needs the host's own (own = its rope_sched) of the same kind (a fixed host plays its own shape)."""
    pos = [r for r in need if r != 0]
    if not pos:
        return True                                        # nothing rotates (a NoPE-only schedule: zero_freq_ok)
    if kind_t and kind_t != kind_h:
        return False
    if len(pos) != 1 or pos[0] < 0:
        return own is not None and tuple(need) == tuple(own) and hd_t <= hd_h
    dt = pos[0]
    if frac_free:
        return dt <= hd_h
    return any(c > 0 and dt <= c and hd_t - dt <= hd_h - c for c in cap)


def zero_freq_room(p):
    """Head dims a (partial) RoPE host can hold at frequency 0 under config-only rules: all of them on a pairs-free
    host (longrope factors) or a fraction-free one (rotary fraction 0), else its pass dims (head_dim - the widest
    table): a fixed full-RoPE host has none (Lean rope_theta_cannot_reach_nope: inv_freq[0] = 1 for any theta)."""
    d = p["dims"]
    hd = int(d.get("head_dim", 0))
    if d.get("rope_pairs_free") or (d.get("rot_frac_free") and _p5("partial")) or not _p5("rot_config"):
        return hd
    cap = [c for c in d.get("rope_cap", d.get("rope_sched", ())) if c > 0]
    return max(0, hd - max(cap, default=0))


def kv_hosts_needed(kt, gt, gh, q):
    """Host KV heads (each with gh query heads, holding up to q target KV heads side by side) that a target of
    kt KV heads with gt query heads each needs: the target's query heads laid end to end, each host head takes
    the longest run it can -- <= gh query heads from <= q distinct target KV heads (a run split across host
    heads replicates that target head's K, V). The reach of a head is monotone in its start, so the greedy run
    is optimal among contiguous layouts. q = 1 is the replication rule kt * ceil(gt / gh)."""
    if kt <= 0 or gt <= 0 or gh <= 0:
        return 0
    if q <= 1:
        return kt * -(-gt // gh)
    y, heads, end = 0, 0, kt * gt
    while y < end:
        y = min(y + gh, (y // gt + q) * gt)
        heads += 1
    return heads


def kv_heads_fit(P, mode="payable", hd_add=None):
    """[i, j]: host P[j] has the KV heads for target P[i] (by count, replicated or PACKED; see _kv_fit_shape). hd_add
    ([i, j], round 2 pass 5): head dims the pair adds to the target's (a sink target's spare zero-frequency slots).
    A stack with several per-layer attention shapes (dims kv_shapes: MiMo-V2.5's sliding layers 8 KV heads x 8 q
    beside its full layers' 4 x 16, Gemma-4's global layers; verify/REVIEW.md) needs EVERY target shape to fit SOME
    host shape (layer alignment is not modelled, as for the rot gate): reflexive, and transitive as each shape's
    fit is (round 2, pass 5).
    Round 3 (per_layer_head_shapes), every mode but capacity: the fit is COUNTED -- a host offers each shape only on
    its own layers (dims kv_shape_n; a single-shape stack on its FU + FT + LU + LT eff layers), and the target's layers
    of each shape need distinct host layers of a fitting shape: Hall's condition over the sets U of target shapes,
    sum of the target's layers in U <= the host's layers whose shape fits some shape in U (the kind_match pattern). A
    matching composes with a matching (each shape's fit is transitive), so the relation stays transitive; on two
    single-shape stacks it is the shape fit plus the attention-layer count kind_match already demands. Each shape's
    fit also needs its head dim (per_layer_head_shapes) and V width (v_head_dim_narrow) on the host shape."""
    import numpy as np
    shapes = [_kv_shape_list(p) for p in P]
    S = max((len(x) for x in shapes), default=1)
    fit = {(a, b): _kv_fit_shape(P, mode, hd_add, shapes, a, b) for a in range(S) for b in range(S)}
    out = np.ones((len(P), len(P)), bool)
    for a in range(S):
        acc = np.zeros((len(P), len(P)), bool)
        for b in range(S):
            acc |= fit[a, b]
        out &= acc
    if mode != "capacity" and _r3("per_layer_head_shapes"):
        from itertools import combinations
        cnt = np.zeros((len(P), S), dtype=np.int64)
        for i, p in enumerate(P):
            c = _kv_counts(p, len(shapes[i]))
            cnt[i, :len(c)] = c
        for r in range(1, S + 1):
            for U in combinations(range(S), r):
                dem = cnt[:, list(U)].sum(1)
                sup = np.zeros((len(P), len(P)), dtype=np.int64)
                for b in range(S):
                    reach = np.zeros((len(P), len(P)), bool)
                    for a in U:
                        reach |= fit[a, b]
                    sup += np.where(reach, cnt[None, :, b], 0)
                out &= sup >= dem[:, None]
    return out


def _kv_shape_list(p):
    """A point's attention shapes as (kv heads, q per kv, head dim, v head dim): kv_shapes (pass-5 3-tuples get v = the
    head dim), else the top-level one (v = dims v_head, else head_dim)."""
    dm = p["dims"]
    sh = [tuple(int(v) for v in s) for s in (dm.get("kv_shapes") or ())]
    if sh:
        return [s + (s[2],) if len(s) < 4 else s[:4] for s in sh]
    hd = int(dm.get("head_dim", 0))
    return [(int(dm.get("kv_heads", 0)), int(dm.get("q_per_kv", 0)), hd, int(dm.get("v_head", 0) or hd))]


def _kv_counts(p, n_shapes):
    """Round 3 (per_layer_head_shapes): the attention layers of each of a point's n_shapes shapes -- dims kv_shape_n,
    else (a single-shape stack, a synthetic point) its FU + FT + LU + LT eff layers spread evenly (_kv_shape_counts)."""
    c = tuple(int(x) for x in (p["dims"].get("kv_shape_n") or ()))
    if len(c) == n_shapes:
        return c
    total = sum(int(_kind_count(p, k, "eff")) for k in ("FU", "FT", "LU", "LT"))
    raw = [total / n_shapes] * n_shapes
    out = [int(r) for r in raw]
    for i in range(total - sum(out)):
        out[i] += 1
    return tuple(out)


def _kv_fit_shape(P, mode, hd_add, shapes, a, b):
    """kv_heads_fit for the targets' a-th and the hosts' b-th attention shape (a point with fewer repeats its last)."""
    import numpy as np
    pick = lambda c, k: np.array([x[min(c, len(x) - 1)][k] for x in shapes], dtype=np.int64)
    hd_tv, hd_hv = pick(a, 2), pick(b, 2)
    hd_t = hd_tv[:, None] + (hd_add if hd_add is not None else 0)

    def dim(k):
        return np.array([p["dims"].get(k, 0) for p in P], dtype=np.int64)
    sa = lambda s: np.array([p["slots"][s] for p in P])
    kvv = sa("kv")
    # KV heads by COUNT, with KV-HEAD PACKING (round 2, pass 2). A target group of g_t query heads on host groups
    # of g_h query heads needs ceil(g_t / g_h) host KV heads (its K, V replicated): host kv_heads >= target
    # kv_heads * ceil(g_t / g_h). PACKING (verify/levers/constructions.py C on branch frontier-levers: 4e-16 in
    # fp64): one host KV head of head_dim D holds m target KV heads of head_dim d side by side when m d <= D --
    # K = [k_a; k_b; ...] (zero-padded), a query of target group a is [q_a; 0; ...] (the host's 1/sqrt(D) scale
    # folded into W_q), V = [v_a; v_b; ...] and o_proj reads group a's block (softmax normalises per QUERY head,
    # so the blocks do not mix: exact without RoPE, Lean KVPack.kvPack_nope); with RoPE the host's table must hold
    # the target's frequencies once per block -- a longrope config on a rope_pairs_free host, rope_fits on m x the
    # target's rotated dims (below). Each block needs
    # at least one of the host head's g_h query heads, and a target group may be split over several host heads
    # (its K, V replicated, as without packing): the count is kv_hosts_needed with q = min(D // d, g_h) blocks per
    # host head, lowered until the RoPE tables hold q blocks. The composition of two packings is a packing (q's
    # multiply, widths and rotated dims too), so the relation stays transitive. Plain (GQA / MQA / MHA) attention
    # on both sides only: an MLA head's keys share one rotary key (KV_ORDER).
    kvh, qpg = pick(a, 0), pick(a, 1)                     # the targets' shape a
    kvh_h, qpg_h = pick(b, 0), pick(b, 1)                 # the hosts' shape b
    have, have_h = (kvh > 0) & (qpg > 0), (kvh_h > 0) & (qpg_h > 0)
    n = len(P)
    # q[i, j]: target KV heads one host KV head may hold side by side (1 = no packing): the width D // d, the host
    # group (each block needs >= 1 of its query heads), plain attention on both sides, and the positional room
    # (mode="capacity" drops every type wall, the MLA one too: packing on every pair)
    plain_kv_h = np.array([mode == "capacity" or all(b != "MLA" for b in branches(v)) for v in kvv])
    q = np.minimum(hd_hv[None, :] // np.maximum(hd_t, 1), qpg_h[None, :])
    vw = _r3("v_head_dim_narrow")
    v_t, v_h = pick(a, 3), pick(b, 3)
    if vw:     # round 3: the V blocks side by side need the host's V width too (zero-padded V columns, o_proj reads 0)
        q = np.minimum(q, v_h[None, :] // np.maximum(v_t[:, None], 1))
    q = np.where(plain_kv_h[:, None] & plain_kv_h[None, :], np.maximum(q, 1), 1)
    if mode != "capacity" and (q > 1).any():
        # RoPE: packed blocks of a RoPE target each rotate with the target's frequencies, so the host's table must
        # REPEAT them, one copy per block (Lean KVPack.kvPack_rope_iff: exact iff every target pair lands on a host
        # pair of the same frequency). The default schedule theta^(-2i/d) has distinct frequencies, so two RoPE KV
        # heads never pack through rope_theta (kvPack_default_not_exact); a host whose rotary takes the rope type
        # from its config (rope_pairs_free) sets one with LONGROPE: inv_freq_i = 1 / (f_i theta^(2i/d)), a factor
        # f_i per pair (short_factor = long_factor, so the length switch is a no-op; attention_factor = 1 or folded
        # into W_q) -- shape-preserving config. Then rope_fits on q x the target's rotated dims. An ALiBi target on a
        # (partial) RoPE host keeps ONE spare pair at inv_freq = 0 beside the q blocks; NoPE / ALiBi / learned-abs
        # targets need only zero frequencies, as their unpacked edges do (kvPack_nope).
        posk = sa("pos")
        alp = (posk == "alibi")[:, None] & np.isin(posk, ("rope", "partial_rope"))[None, :]
        q = np.where(alp, np.maximum(np.minimum(q, (hd_hv[None, :] - 2) // np.maximum(hd_t, 1)), 1), q)
        rotp = np.isin(posk, list(ROT_TARGETS))[:, None] & np.isin(posk, list(ROT_HOSTS))[None, :]
        q = np.where(rotp & (dim("rope_pairs_free") == 0)[None, :] & bool(_cr("kv_rope_pack")), 1, q)
        if _wa("wa_pack_zero_room"):
            # wall audit (wa_pack_zero_room, soundness): a position-free target's packed blocks (NoPE, ALiBi's content
            # pairs, a learned-abs target's attention) each sit at frequency 0 on a (partial) RoPE host (kvPack_nope
            # needs them unrotated): q blocks need q hd_t zero-frequency dims (zero_freq_room; ALiBi + its one small-angle
            # pair), as the unpacked edge needs hd_t (the pos slot) -- unless the host's linear-scaling lever sends every
            # angle to 0 (nope_rope_limit, payable / mixgap)
            zr = np.array([zero_freq_room(p) for p in P], dtype=np.int64)
            flat = np.isin(posk, ("nope", "alibi", "learned_abs"))[:, None] & np.isin(posk, list(ROT_HOSTS))[None, :]
            lin = ((dim("rope_lin") > 0) & (mode not in ("capacity", "exact")) & _r3("nope_rope_limit"))[None, :]
            room = np.where((posk == "alibi")[:, None], zr[None, :] - 2, zr[None, :]) // np.maximum(hd_t, 1)
            cap_q = np.where((posk == "nope")[:, None] & lin, q, np.minimum(q, np.maximum(room, 1)))
            q = np.where(flat, cap_q, q)
        # (round 3, d_transitivity_rot_gate: a fixed target with cap != sched stands for its cap, as in the rot gate)
        need_s = [host_rope_cap(p) if (p["dims"].get("rope_pairs_free")
                                       or (p["dims"].get("rot_frac_free") and _p5("partial"))
                                       or (_r3("d_transitivity_rot_gate")
                                           and host_rope_cap(p) != tuple(p["dims"].get("rope_sched", ()))))
                  else tuple(p["dims"].get("rope_sched", ())) for p in P]     # a target's class (round 2, pass 5)
        cap_s = [host_rope_cap(p) for p in P]
        memo = {}
        for i, j in zip(*np.nonzero(rotp & (q > 1))):
            key = (need_s[i], cap_s[j], int(q[i, j]))
            if key not in memo:
                m = key[2]
                while m > 1 and _cr("kv_rope_pack") and not rope_fits(tuple(m * x for x in key[0]), key[1]):
                    m -= 1
                memo[key] = m
            q[i, j] = memo[key]
    # host KV heads needed (kv_hosts_needed), per distinct (target heads, target group, host group, q)
    # (one int64 key per pair: kt < 2^20, gt / gh / q < 2^14 -- a 1-D unique, not a row-wise one)
    keys = ((np.minimum(kvh, (1 << 20) - 1)[:, None] << 42) | (np.minimum(qpg, (1 << 14) - 1)[:, None] << 28)
            | (np.minimum(qpg_h, (1 << 14) - 1)[None, :] << 14) | np.minimum(q, (1 << 14) - 1)).reshape(-1)
    uk, inv = np.unique(keys, return_inverse=True)
    m14 = (1 << 14) - 1
    needed = np.array([kv_hosts_needed(int(k >> 42), int((k >> 28) & m14), int((k >> 14) & m14), int(k & m14))
                       for k in uk], dtype=np.int64)[inv.reshape(-1)].reshape(n, n)
    ok = kvh_h[None, :] >= needed
    # round 3: a shape's layers need host heads as wide (per_layer_head_shapes: per shape, not only the widest layer)
    # and V heads as wide (v_head_dim_narrow; Lean qknorm_pad / zeroed_sublayer: zero-padded head dims, o_proj reads 0)
    if _r3("per_layer_head_shapes"):
        ok &= hd_hv[None, :] >= hd_t
    if vw:
        ok &= v_h[None, :] >= v_t[:, None]
    return ~(have[:, None] & have_h[None, :]) | ok


def _r3_masks(P, fts, eps_h, hidden):
    """Round 3, batch 1: the classification dims' pair rules ({check name: [i, j] allowed}, target i on host j; every
    mode but capacity). Each only removes pairs the feature / slot rules allowed (walls), except where noted with the
    theorem that pays it:
      embedding_norm   a host's always-on embedding norm plays only targets whose own kind it nests (EMB_NEST: equal,
                       or a weighted / gated RMS set to the weight-free one -- Lean qkn_gain_one, out_gate_off); a
                       target's norm on a host without a nesting one is a wall unless EMB_FOLD_UNTIED (the fold of an
                       untied table needs theorem emb_fold_untied). The universal host's value is a branch set
                       ("|ln|rms": its switchable UNorm emb_norm, elephantine/universal.py from_bloom)
      grouped_rmsnorm  g_h >= g_t groups and group size hidden_h / g_h >= hidden_t / g_t (each target group in its own
                       host group, zero-padded rows scaled: Lean rmsNorm_pad_scaled; unused groups are 0); a grouped host
                       plays a full-width target only within one group
      gated_prenorm_lowrank  between gated-norm points: host rank >= target rank, host sites >= target sites (Lean
                       lowrank_pad; a host site the target lacks is switched off, out_gate_off)
      norm_placement_keys_unread  a post_norm_attn target the host lacks (a full sandwich host) needs the host's eps
                       lever to switch its MLP post-norm off (payable only: Lean postNorm_layer_bound)
      factorized_embedding_proj  a factorized target (OPT word_embed_proj_dim) needs a factorized host at least as wide
                       (zero-padded E, P_in, P_out) or the universal host (emb_fold: from_opt folds project_in /
                       project_out into its own tables, run against the real opt-350m in tests/test_stock_loaders.py);
                       a factorized host plays only TIED targets with hidden <= its rank (its
                       one table E feeds both P_in and P_out; an untied target on it needs theorem: two tables in one
                       vocab x wpd table, 2 hidden_t <= wpd)"""
    import numpy as np
    n = len(P)
    out = {}
    D = lambda k, z: [p["dims"].get(k, z) for p in P]
    tied = np.array(["tied_emb" in set(f.lstrip("*").split("+")) for f in fts])
    if _r3("embedding_norm"):
        en = D("emb_norm", "")
        uk = sorted(set(en))
        # (a host value may be a branch set: the universal host's switchable UNorm emb_norm, "|ln|rms")
        nest = np.array([[any(a in EMB_NEST.get(c, (c,)) for c in branches(b)) for b in uk] for a in uk])
        ei = np.array([uk.index(x) for x in en])
        m = nest[np.ix_(ei, ei)]
        if EMB_FOLD_UNTIED:
            has = np.array([bool(x) for x in en])
            m = m | (has[:, None] & ~has[None, :] & ~(tied[:, None] & tied[None, :]))
        out["embedding_norm"] = m
    if _r3("grouped_rmsnorm"):
        # (a host value may be a branch set "1|2|4" -- only the universal host's point, minimal_emulator.host_point:
        # its UNorm takes any group count, so it plays a target through the best of its listed counts; every ordinary
        # point has one int and the rule is unchanged)
        raw = D("norm_groups", 1)
        sets = [j for j, v in enumerate(raw) if isinstance(v, str)]
        g = np.array([1 if isinstance(v, str) else v for v in raw], dtype=np.int64)
        g = np.where(g > 0, g, 1)
        s = hidden // g
        grouped = (g[:, None] > 1) | (g[None, :] > 1)
        m = ~grouped | ((g[None, :] >= g[:, None]) & (s[None, :] >= s[:, None]))
        for j in sets:
            col = np.zeros(n, dtype=bool)
            for gh in (max(int(x), 1) for x in branches(raw[j])):
                sh = hidden[j] // gh
                col |= ((g <= 1) & (gh <= 1)) | ((gh >= g) & (sh >= s))
            m[:, j] = col
        out["grouped_rmsnorm"] = m
    if _r3("gated_prenorm_lowrank"):
        r = np.array(D("gnorm_rank", 0), dtype=np.int64)
        st = [frozenset(filter(None, str(x).split("+"))) for x in D("gnorm_sites", "")]
        both = (r[:, None] > 0) & (r[None, :] > 0)
        cover = np.array([[a <= b for b in st] for a in st])
        out["gated_norm rank / sites"] = ~both | ((r[None, :] >= r[:, None]) & cover)
    if _r3("norm_placement_keys_unread"):
        pna = np.array(["post_norm_attn" in set(f.lstrip("*").split("+")) for f in fts])
        out["post_norm_attn lever"] = ~(pna[:, None] & ~pna[None, :]) | eps_h[None, :]
    if _r3("factorized_embedding_proj"):
        er = np.array(D("emb_rank", 0), dtype=np.int64)
        fold = np.array(D("emb_fold", 0), dtype=np.int64) > 0
        ft, fh = (er > 0)[:, None], (er > 0)[None, :]
        out["factorized embedding"] = ((~ft | (fh & (er[None, :] >= er[:, None])) | fold[None, :])
                                       & (~fh | ft | (tied[:, None] & (er[None, :] >= hidden[:, None]))))
    out.update(_r3_masks_2d(P, hidden))
    out.update(_r3_masks_4b(P, fts))
    out.update(_r3_masks_nonlm(P, fts))
    return out


def _r3_masks_4b(P, fts):
    """Round 3, batch 4b pair rules (walls: each only removes pairs; target i on host j):
      attn_res_keys_unread  AttnRes <- AttnRes only at an equal source partition / window (attn_res_sig, _sig_covers;
                       the universal host's Block / Full AttnRes branches "1/0|-1/-1": from_gpts3 / from_alice /
                       from_kimi_linear)
      ngram_memory_unread / qwen4_ple_reads_ngrams  an n-gram memory target on a host of its kind only at equal table
                       geometry (engram_sig; the aligned-layer superset the spec allows needs the layer alignment the
                       relation does not model); the universal host's UEngram takes the target's geometry ("*")
      value_emb_keys_unread  value_emb <- value_emb only at an equal kind (ve_kind; ve_layer <- ve_shared needs theorem
                       ve_shared_per_layer)
      cfrd_film_norm_shift  a SOUNDNESS wall: a host whose biases are one shared norm shift (bias_shift, CFRD) plays no
                       target with independent biases (bias_qkv / bias_out) but its own kind -- a +1-channel construction
                       (the shift on a spare channel is a constant after the norm) is uncharged and has no theorem"""
    import numpy as np
    out = {}
    for spec, key, name in (("attn_res_keys_unread", "attn_res_sig", "attn_res partition"),
                            ("ngram_memory_unread", "engram_sig", "engram geometry"),
                            ("value_emb_keys_unread", "ve_kind", "value_emb kind")):
        if not (_r3(spec) or (key == "engram_sig" and _r3("qwen4_ple_reads_ngrams"))):
            continue
        sv = [str(p["dims"].get(key, "") or "") for p in P]
        uk = sorted(set(sv))
        cov = np.array([[_sig_covers(a, b) for b in uk] for a in uk])
        ci = np.array([uk.index(x) for x in sv])
        out[name] = cov[np.ix_(ci, ci)]
    if _r3("cfrd_film_norm_shift"):
        bs = np.array([int(p["dims"].get("bias_shift", 0) or 0) > 0 for p in P])
        fb = np.array([bool({"bias_qkv", "bias_out"} & set(f.lstrip("*").split("+"))) for f in fts])
        out["bias shift host"] = ~(bs[None, :] & ~bs[:, None] & fb[:, None])
    return out


def _r3_masks_nonlm(P, fts):
    """Round 3 (nonlm_heads) pair rules on the head / input slots (target i on host j; every mode but capacity). The
    slots' orders (HEAD_ORDER) let a score / embedding target onto an LM host and keep every other kind change a wall;
    this adds the readout rule's conditions (walls: each only removes pairs):
      readout   a target with a non-LM head is played EITHER through an LM host's lm_head (score; an embedding only
                when its readout is a fixed read, readout_sig in FIXED_READS -- mean pooling / L2 are not; a head bias
                needs the host's bias_out; an untied host: FEATURES tied_emb, host side a wall, and _feats drops
                tied_emb from non-LM heads) OR by a host head of its own kind with the same readout (readout_sig /
                flow_sig equal, or the universal host's "*") and widths readout_w / readout_ff >= the target's, a head
                bias only from a biased host head (a host bias is set to 0 for a bias-free target)
      patches   a patch input on a patch host only at the same patch embedding (patch_sig, or "*")"""
    import numpy as np
    if not _r3("nonlm_heads"):
        return {}
    D = lambda k, z: [p["dims"].get(k, z) for p in P]
    head = [p["slots"].get("head", "lm") for p in P]
    hb = [set(branches(h)) for h in head]
    I = lambda k: np.array([int(x or 0) for x in D(k, 0)], dtype=np.int64)
    rw, rf, rb = I("readout_w"), I("readout_ff"), I("readout_bias")
    sig = [str(x or "") for x in D("readout_sig", "")]
    fsig = [str(x or "") for x in D("flow_sig", "")]
    psig = [str(x or "") for x in D("patch_sig", "")]
    bias_out = np.array(["bias_out" in set(f.lstrip("*").split("+")) for f in fts])

    def pair(fn, vals):
        uk = sorted(set(vals))
        cm = np.array([[fn(a, b) for b in uk] for a in uk])
        ci = np.array([uk.index(x) for x in vals])
        return cm[np.ix_(ci, ci)]
    cov = _sig_covers               # (two heads of one kind carry both signatures or neither: "" never meets a value)
    kinds = sorted({k for s in hb for k in s})
    has = {k: np.array([k in s for s in hb]) for k in kinds}
    tk = np.array(head)
    non_lm = tk != "lm"
    fixed = np.array([s in FIXED_READS for s in sig])
    via_lm = (np.isin(tk, ("score", "embedding")) & ((tk != "embedding") | fixed))[:, None] \
        & has.get("lm", np.zeros(len(P), bool))[None, :] & (~(rb > 0)[:, None] | bias_out[None, :])
    own = np.zeros((len(P), len(P)), bool)
    for k in kinds:
        own |= (tk == k)[:, None] & has[k][None, :]
    via_own = (own & pair(cov, sig) & pair(cov, fsig) & (rw[None, :] >= rw[:, None]) & (rf[None, :] >= rf[:, None])
               & (rb[None, :] >= rb[:, None]))
    out = {"readout": ~non_lm[:, None] | via_lm | via_own}
    patches = np.array([p["slots"].get("input", "tokens") == "patches" for p in P])
    out["patch input"] = ~patches[:, None] | pair(cov, psig)
    return out


def _r3_masks_2d(P, hidden):
    """Round 3, batch 2d pair rules (walls: each only removes pairs; target i on host j):
      grouped_lowrank_oproj  a grouped host (o_groups g_h, rank o_rank r_h per group: o = W_b blockdiag(A) attn) plays a
                       target whose o-map fits: m = ceil(g_t / g_h) target groups (a dense target: 1 group of all its
                       heads) in each host group, m ceil(H_t / g_t) <= floor(H_h / g_h) heads, and r_h >= the rank those
                       groups need, min(m r_t, m ceil(H_t / g_t) d_t, hidden_t) (a dense target's r_t unbounded; Lean
                       fullrank_plays_lowrank / lowrank_pad: zero / identity padding). A dense host plays any target
                       (W_o := W_b blockdiag(A)). Transitive: ceil(a / c) <= ceil(a / b) ceil(b / c), and the rank bound
                       is capped by widths the qheads / head_dim / hidden checks order.
      step3_mfa_query_latent_norm  a normed query latent (Step-3 share_q_dim) only on an equal one: wider latents (the
                       rmsNorm_rep multiple, the eps lever) and a latent host for a plain target (the eps lever: the
                       q-latent instance of leverAttn_layer_bound, unmeasured) are not credited
      canon_conv_layers  a Canon target on a Canon host: equal positions and kernel (_canon_covers; the spec's cover --
                       target positions and kernel within the host's, the rest 0 -- needs theorem canon_pad: CANON_PAD)
      moa_routed_attention_heads  MoA <- MoA: equal experts and top-k (a wider router needs the routing rules' extra-
                       expert argument for a bias-free top-k softmax router: not credited)"""
    import numpy as np
    out = {}
    D = lambda k, z=0: np.array([p["dims"].get(k, z) for p in P], dtype=np.int64)
    if _r3("grouped_lowrank_oproj"):
        g, r = D("o_groups"), D("o_rank")
        H, hd = np.maximum(D("qheads"), 1), D("head_dim")
        gt = np.where(g > 0, g, 1)[:, None]
        gh = np.maximum(g, 1)[None, :]
        m = -(-gt // gh)
        ht = -(-H[:, None] // gt)
        hh = H[None, :] // gh
        big = np.int64(1) << 40
        rt = np.where(g > 0, r, big)[:, None]
        need = np.minimum(np.minimum(m * rt, m * ht * hd[:, None]), hidden[:, None])
        out["o_proj groups"] = (g[None, :] == 0) | ((m * ht <= hh) & (r[None, :] >= need))
        if _r3("o_groups_reflexive"):
            # round 3 (reflexivity): with qheads not a multiple of o_groups, ceil(H / g) > floor(H / g) refused a shape
            # its own o-map (the codes assert divisibility; Z3 found the realistic shape). The identical o-map (same
            # groups, rank, heads, head dim, hidden) is the identity construction; transitive with the rule above (an
            # identical endpoint gives the other pair's own formula)
            same = ((g[:, None] == g[None, :]) & (r[:, None] == r[None, :]) & (H[:, None] == H[None, :])
                    & (hd[:, None] == hd[None, :]) & (hidden[:, None] == hidden[None, :]))
            out["o_proj groups"] = out["o_proj groups"] | same
    if _r3("step3_mfa_query_latent_norm"):
        ql = D("q_lat_norm")
        out["q latent norm"] = ql[:, None] == ql[None, :]
    if _r3("canon_conv_layers"):
        cn = [str(p["dims"].get("canon", "") or "") for p in P]
        uk = sorted(set(cn))
        cov = np.array([[(not a) or (not b) or _canon_covers(a, b) for b in uk] for a in uk])
        ci = np.array([uk.index(x) for x in cn])
        out["canon cover"] = cov[np.ix_(ci, ci)]
    if _r3("moa_routed_attention_heads"):
        me, mk = D("moa_e"), D("moa_k")
        both = (me[:, None] > 0) & (me[None, :] > 0)
        out["moa routing"] = ~both | ((me[:, None] == me[None, :]) & (mk[:, None] == mk[None, :]))
    return out


def emul_matrix(P, exact=None, capacity_only=False, mode=None, explain=None):
    """ok[i,j] = P[j] emulates P[i]  (i <= j). Vectorized; scales to thousands.

    mode="payable"  (default; the HEADLINE) capacity + PAYABLE type walls: a slot where host and
        target differ is crossed only by an edge that has a currency -- exact, a floor / regime (the
        activation floors <= ACT_FLOOR_MAX, softcap, qk-norm), or a limit (ALiBi <- RoPE, ReLU <-
        swish); a difference with no currency blocks. The softmax <- SSM / linear / hybrid near-gap
        is refused (MIXER_GAP_PAYABLE: measured, its constructions need heads / depth / a norm change
        the edge does not charge), and so is the CLA carry (CLA_CARRY_PAYABLE: a non-sharing host
        for a KV-sharing target leaves a ~6e-2 pre-norm floor), and so is every qk-norm / softcap regime
        direction measured not payable (REGIME_PAYABLE: a qk-norm variant needs the same variant or one that
        contains it, an attention-capped target an attention-capped host). The ZERO_EDGES (a hybrid /
        sliding-window / KV-sharing host for a plain softmax target) charge depth: only the host's layers of
        the target's kind count. KV heads may be PACKED (kv_heads_fit). ALiBi hosts need the target's slopes
        (alibi_nests) and play NoPE only where a config value switches the bias off.
        A RoPE-family target on a RoPE-family host (full or partial) is ROT-GATED (every mode but
        capacity): the target's rotating pairs, and its NoPE layers' zero-frequency slots, must fit the
        host's inv_freq tables, rope_fits(dims rope_sched, rope_cap) (hf/partial_rope_host.py: the exact
        construction).
        MoE FFNs (round 2, pass 3: _moe_ok / moe_why on dims moe_sig): MoE <- MoE by expert duplication (top-k a
        multiple m, E_h >= m E_t, widths, renorm / score / group shape per family, extras sunk), dense / MoE layer
        kinds counted, a dense host only for a top-1 unit-gate target (+E channels, +2E(E-1) units, 2x depth).
        MLA <- MLA latents an integer multiple of the target's (hard-coded latent-norm eps); DSA is dense attention.
        Recurrent layers (round 2, pass 4: _rec_ok on dims RC_DIMS / rc_rule): the host's recurrent layers of the
        target's family at least as large per the family's construction (Mamba-2 grouped norms as whole-group copies,
        GDN / KDA key head dims equal), Zamba2 shared blocks a multiple of the target's.
        Round 2, pass 5 (the block above kv_hosts_needed): a host sets the config values its family's code reads --
        its sliding window >= the context (WINDOW_FREE_TYPES), its rotary fraction (ROT_FRAC_FREE_TYPES), its softcap
        values, its q / k norm flag; RoPE frequencies only on a pairs-free rotary (A3: fixed_rot_fits, zero_freq_room);
        the eps lever switches its qk-norm / sandwich post-norms / v-norm / MLA latent norm off and relaxes the MLA /
        Mamba-2 / GDN / KDA multiples (payable, a limit); the start-token rules (learned-abs, sink), activation width
        (FFN x k), GeLU erf <-> tanh, looped hosts (only a plain one, charged), the pre-norm compensators (measured
        families only; round 3, batch 3b: removed, spot_comp_removed -- P5_REMOVED_BY), and the walls: a swigluoai target needs a host up bias, a LayerNorm-beta target on an RMSNorm
        host an lm_head bias, ALiBi on a learned-abs host none. The per-pair charges are add_h / add_l / add_hd /
        kmul / f_add / f_min; P5_OFF switches a port off (the leave-one-out table); explain= names the refusing checks.
    mode="mixgap"   as payable, but the softmax <- SSM / linear / hybrid near-gap and the CLA carry
        granted free (the pre-measurement headline; the regime slots still follow REGIME_PAYABLE).
    mode="nomix"    legacy alias of payable (the near-gap and the CLA carry refused).
    mode="exact"    every crossed edge must be exact (no floor / limit / regime / gap).
    mode="capacity" the secondary "if every type wall were free": only the one-way gen wall
        (AR <= diffusion) is kept, every other slot ignored -- pure capacity dominance.
    Legacy flags: exact=True / "nomix" / False, capacity_only=True (see _mode).
    explain: a dict to fill with {check name: [i, j] mask of the pairs that check refuses} (graph/frontier.blockers)."""
    import numpy as np
    mode = _mode(mode, exact, capacity_only)
    strict = mode == "exact"
    n = len(P)

    def _why(name, m):                 # record which check refuses which pairs (explain), pass the mask through
        if explain is not None:
            m = np.broadcast_to(np.asarray(m, bool), (n, n))
            explain[name] = explain.get(name, np.zeros((n, n), bool)) | ~m
        return m

    def dim(k, fallback=None):
        return np.array([p["dims"].get(k, p["dims"].get(fallback, 0) if fallback else 0) for p in P],
                        dtype=np.int64)
    hidden = dim("hidden")
    pl, el = dim("param_layers"), dim("eff_layers")   # distinct layers / unrolled depth
    kv, q, fT, fA = dim("kv_cap"), dim("q_cap"), dim("ffn_total"), dim("ffn_active")
    qh, hd = dim("qheads"), dim("head_dim")
    sa = lambda s: np.array([p["slots"].get(s, SLOT_DEFAULT.get(s)) for p in P])
    kvv, ffn = sa("kv"), sa("ffn")
    frank = np.array([FFN_RANK.get(f, max(FFN_RANK.values())) for f in ffn])

    def cat_ok(vals, slot, strict_slot=False):   # [i,j] = target vals_i hosted by vals_j (i <= j)
        uniq = sorted(set(vals.tolist()))
        idx = {u: i for i, u in enumerate(uniq)}
        # the regime slots (qk-norm / softcap) follow REGIME_PAYABLE in every mode but exact (and capacity)
        reach_ = (regime_reaches if slot in REGIME_SLOTS and not strict_slot else
                  lambda sl, a, b: slot_reaches(sl, a, b, strict_slot))
        cm = np.array([[reach_(slot, a, b) for b in uniq] for a in uniq])
        ci = np.array([idx[v] for v in vals])
        return cm[np.ix_(ci, ci)]

    ok = np.ones((n, n), bool)
    # round 2, pass 5: the per-pair charges of the rules that cross a wall at a price (zero / one where no rule is
    # used): extra hidden (after the norm / carry width factor), extra full-attention layers, extra head dims (the
    # sink's spare slots), the FFN width multiple k (activation width), FFN units added per layer, an FFN floor
    add_h = np.zeros((n, n), np.int64)
    add_l = np.zeros((n, n), np.int64)
    add_hd = np.zeros((n, n), np.int64)
    kmul = np.ones((n, n), np.int64)
    f_add = np.zeros((n, n), np.int64)
    f_min = np.zeros((n, n), np.int64)
    flag = lambda k: dim(k) > 0
    ln_comp = np.zeros((n, n), bool)                  # pairs crossed by the LayerNorm-fold compensator
    eps_h = flag("eps_lever") & (mode not in ("capacity", "exact"))     # the eps lever: payable (a limit)
    # round 3 (nope_rope_limit): the linear-scaling lever, payable (a limit) like the eps lever
    lin_h = flag("rope_lin") & (mode not in ("capacity", "exact")) & _r3("nope_rope_limit")
    if mode == "capacity":
        ok &= _why(f"gen", cat_ok(sa("gen"), "gen"))  # AR <= diffusion (mask), one-way
        usable_e, usable_p = np.broadcast_to(el[None, :], (n, n)), np.broadcast_to(pl[None, :], (n, n))
        host_full = None
    else:
        lever = not strict                                # the payable rules: limits, measured rescues
        pos = sa("pos")
        rot_t, rot_h = np.isin(pos, list(ROT_TARGETS)), np.isin(pos, list(ROT_HOSTS))
        pf, ff = flag("rope_pairs_free"), flag("rot_frac_free") & _p5("partial")
        zroom = np.array([zero_freq_room(p) for p in P], dtype=np.int64)     # host dims at frequency 0
        rot_need = np.array([max([r for r in p["dims"].get("rope_sched", ()) if r > 0], default=0) for p in P],
                            dtype=np.int64)
        norm_v, act_v = sa("norm"), sa("act")
        gated_h = frank >= FFN_RANK["glu"]
        full_any = (dim("full_eff", "eff_layers") > 0) | (flag("win_free") & _p5("window"))
        for slot in CAT_SLOTS:
            m = cat_ok(sa(slot), slot, strict or _regime_strict(slot))
            v = sa(slot)
            if slot == "qknorm":
                # a host switches its q / k norm OFF for a target without one: a config flag (exact), or the eps lever
                # on a gained RMS variant (payable, a limit)
                lev_qk = np.array([any(b in EPS_LEVER_QKNORM for b in branches(x)) for x in v])
                m |= (v == "no")[:, None] & ((flag("qk_switch") & _p5("qk_switch"))
                                             | (lev_qk & eps_h & _p5("eps_qknorm")))[None, :]
            elif slot == "softcap":
                # softcap values are config where the code reads them: a null cap is no cap, exactly
                sf = flag("softcap_free")[None, :] & _p5("softcap")
                m |= sf & (((v == "no")[:, None] & np.isin(v, ("attn", "final"))[None, :])
                           | ((v == "final")[:, None] & (v == "attn")[None, :]))
            elif slot == "norm":
                # the LayerNorm -> RMSNorm fold with beta != 0 reads beta through the final norm: exact only with an
                # lm_head bias (Lean condition 8); else the 1-D RMS compensator (payable, measured families only)
                ln_t = (v == "LayerNorm") & flag("ln_beta")
                rms_h = np.array(["LayerNorm" not in branches(x) for x in v])
                fold = ln_t[:, None] & rms_h[None, :] & m & _p5("ln_beta")
                kc = np.where(gated_h, COMPENSATOR_K["glu"],
                              np.where(flag("ffn_bias") & np.isin(act_v, list(BOS_ODD_LINEAR_ACTS)),
                                       COMPENSATOR_K["relu"], 0))
                # round 3, batch 3b (spot_comp_removed, verify/spot/SPOT.md): the compensator's construction is measured only by
                # the scripts of verify/prenorm_rescue/ (no host construction or test builds it), and credits no edge on the snapshot (every
                # LayerNorm target on an RMSNorm host without an lm_head bias is beta-free: OLMo-1, MPT, replit -- the
                # exact fold, catalog ref_norm_ln_fold_beta_free): the credit is off (the code path kept)
                comp = (fold & ~flag("lm_head_bias")[None, :] & (lever and p5_port_live("comp_ln"))
                        & flag("rms_ok")[:, None] & (kc > 0)[None, :])
                # fr-resnorm (fr_ln_beta_readout): the final beta by a DOMINATED READOUT -- one more host layer whose
                # MLP (a constant gate: its gate bias; or identity pairs phi(v) - phi(-v) = v on an ungated odd-linear
                # host) writes M [gamma x_hat + beta; kappa] from its pre-norm output, the target's final LayerNorm,
                # into the stream and a pad channel; the final RMSNorm then divides by ~ M kappa / sqrt(D), a constant,
                # and lm_head = (kappa / sqrt(D)) [U; 0] reads y . U (Lean dominated_readout_limit, pad_ratio_limit:
                # a limit; the logits are a positive multiple of the target's at every kappa, pos_scale_le_iff). Needs
                # the host's FFN-input and output biases (the betas inside the stack fold into them as before), an
                # untied host head and a dense FFN; costs +1 layer, +1 hidden, d (gated) / 2 d (ungated) FFN units
                ro = np.zeros_like(fold)
                if lever and _r3("fr_ln_beta_readout") and _p5("ln_beta_readout"):
                    h_feats = [set(p["slots"].get("feats", "").split("+")) for p in P]
                    ro_h = np.array([("tied_emb" not in f) and ("bias_out" in f) for f in h_feats]) & flag("ffn_bias") \
                        & np.isin(ffn, ("glu", "vanilla"))
                    k_ro = np.where(gated_h, 1, np.where(np.isin(act_v, list(BOS_ODD_LINEAR_ACTS)), 2, 0))
                    ro = fold & ~flag("lm_head_bias")[None, :] & ~comp & (ro_h & (k_ro > 0))[None, :]
                    add_h += np.where(ro, 1, 0)
                    add_l += np.where(ro, 1, 0)
                    f_min = np.maximum(f_min, np.where(ro, k_ro[None, :] * hidden[:, None], 0))
                m &= ~fold | flag("lm_head_bias")[None, :] | comp | ro
                ln_comp = comp
                add_h += np.where(comp, 2, 0)
                add_l += np.where(comp, 1, 0)
                f_min = np.maximum(f_min, np.where(comp, kc[None, :], 0))
            elif slot == "pos":
                # A3: a target needing zero-frequency head dims on a (partial) RoPE host gets them only where the host's
                # config sets them (zero_freq_room); ALiBi <- RoPE also needs its small-angle pair set (pairs-free, or a
                # fraction-free / fixed host using a low pair with a large theta beside the zero dims)
                if not _p5("alibi_labs"):          # (pass 4: alibi <- learned_abs by a reserved channel)
                    m |= (v == "alibi")[:, None] & (v == "learned_abs")[None, :]
                nz = np.isin(v, ("nope", "alibi"))[:, None] & rot_h[None, :]
                # round 3 (nope_rope_limit, a host config lever, a LIMIT): a NoPE target on a RoPE host whose code reads
                # linear rope scaling for every rotating table (rope_lin): factor -> inf, every angle -> 0 (Lean
                # rope_scale_limit_identity) -- payable / mixgap only (lin_h is off in exact and capacity); elementwise
                # nope_rope_limit_slot
                lin_nope = (v == "nope")[:, None] & rot_h[None, :] & lin_h[None, :]
                m &= ~nz | (zroom[None, :] >= hd[:, None]) | lin_nope
                al = (v == "alibi")[:, None] & rot_h[None, :]
                m &= ~al | pf[None, :] | (zroom[None, :] >= hd[:, None] + np.where(ff, 4, 2)[None, :])
                # fr-positions (fp_sinusoid_table): a FIXED sinusoid table is a table -- a learned-abs host loads its rows
                # (Lean sinusoid_sub_learned_abs: the rows at the host's index offset, the target's embedding scale folded
                # into the token table), exact; the table length is context, unpriced as for learned_abs <- learned_abs.
                # A sinusoidal host computes ONE table (its formula at its width): it plays a sinusoidal target only with
                # the same one (sin_sig; a wider or other-formula sinusoid differs at position 1, Lean
                # sinusoid_width_differs), never a learned one ("*" <- sinusoidal stays a wall)
                sin_t = (v == "sinusoidal") & _r3("fp_sinusoid_table")
                if sin_t.any():
                    sig = np.array([str(p["dims"].get("sin_sig", "")) for p in P])
                    m |= sin_t[:, None] & (v == "learned_abs")[None, :] & (dim("pos_table")[None, :]
                                                                            >= dim("pos_table")[:, None])
                    if _r3("fp_sinusoid_same_table"):
                        m &= ~(sin_t[:, None] & (v == "sinusoidal")[None, :]) | (
                            (sig[:, None] == sig[None, :]) & (dim("pos_table")[None, :] >= dim("pos_table")[:, None]))
                if _r3("fp_table_length"):
                    # fr-positions: a learned table plays another only over the rows it has -- host pos_table >= the
                    # target's (a table of unknown length, 0, hosts only unknown ones). Forced by transitivity once a
                    # fixed table rides on learned hosts: the start-token rule charges T + 1 units of the TARGET's
                    # table, so X <= B <= C (C via the start-token rule) needs T_X <= T_B (sshleifer/tiny-ctrl's
                    # 50000 rows <= gpt2-hindi's 1024 <= a RoPE host: 36 violations without it)
                    lab = (v == "learned_abs")[:, None] & (v == "learned_abs")[None, :]
                    m &= ~lab | (dim("pos_table")[None, :] >= dim("pos_table")[:, None])
                # the start-token rule: a learned-abs target on a position-free / RoPE host (exact)
                bos_h = (np.isin(v, ("nope", "rope", "partial_rope"))
                         & np.isin(norm_v, list(BOS_EXTRA_HIDDEN)) & np.isin(act_v, list(BOS_KNOT_ACTS))
                         & (gated_h | np.isin(act_v, list(BOS_ODD_LINEAR_ACTS)))
                         & np.isin(sa("mixer"), ("softmax", "hybrid")) & full_any)
                tab_t = (v == "learned_abs") | sin_t           # (fp_sinusoid_table: a fixed table is played as one)
                bos = (tab_t[:, None] & bos_h[None, :] & ~m & _p5("bos_labs")
                       & (~rot_h[None, :] | (zroom[None, :] >= hd[:, None])))
                if _r3("max_context_unknown"):
                    # round 3: the rule's T + 1 units need the table's length; an unknown one (pos_table 0) pays nothing
                    bos &= (dim("pos_table") > 0)[:, None]
                m |= bos
                add_h += np.where(bos, np.array([BOS_EXTRA_HIDDEN.get(x, 5) for x in norm_v])[None, :], 0) * _cr("bos")
                add_l += np.where(bos, BOS_EXTRA_LAYERS, 0) * _cr("bos")
                units = (dim("pos_table")[:, None] + 1
                         + np.where(gated_h, BOS_ERASERS[True], BOS_ERASERS[False])[None, :])
                f_min = np.maximum(f_min, np.where(bos, units, 0) * _cr("bos"))
            elif slot == "sink":
                # the start-token rule: a sink target on a sink-free host with a q bias, a spare zero-frequency slot
                # (+2 head dims rotating, +1 flat) and one reserved channel (exact)
                no_sink = np.array([b == "no" for b in v])
                spare = np.where(rot_h, SINK_SPARE_ROT, SINK_SPARE_FLAT)
                sk = ((v == "yes")[:, None] & no_sink[None, :] & flag("q_bias")[None, :] & ~m & _p5("bos_sink")
                      & (hd[None, :] >= hd[:, None] + spare[None, :])
                      & (~rot_h[None, :] | (zroom[None, :] >= hd[:, None] - rot_need[:, None] + spare[None, :])))
                m |= sk
                add_h += np.where(sk, SINK_EXTRA_HIDDEN, 0) * _cr("sink")
                add_hd += np.where(sk, spare[None, :], 0) * _cr("sink")
            elif slot == "act":
                # a swigluoai target's (up + 1) needs a host up bias unless the host runs swigluoai itself
                oai_h = np.array(["swigluoai" in branches(x) for x in v])
                no_up = (v == "swigluoai")[:, None] & ~oai_h[None, :] & ~flag("ffn_bias")[None, :] & _p5("oai_bias")
                m &= ~no_up
                if _r3("w3_relu_oai_bias"):
                    # round 3, batch 3b (verify/r3_notes.md W3-1): ReLU <- swigluoai is the limit of Lean
                    # relu_sub_swigluoai -- gate row c z, up row a z with an up BIAS -1 cancelling the unit's (up + 1),
                    # bounded inputs |a z| <= 6 (a magnitude regime: ACT_LIMIT); the scaled-gate route is refuted (the
                    # gate clamp caps it at K / c -> 0, swigluoai_scaled_gate_vanishes). A host without an up bias is a
                    # wall (no act-width fit re-crosses it)
                    ro = (v == "ReLU")[:, None] & (v == "swigluoai")[None, :] & ~flag("bias_up")[None, :]
                    m &= ~ro
                    no_up = no_up | ro
                gelu_cells = (((v == "GeLU")[:, None] & (v == "GeLU_tanh")[None, :])
                              | ((v == "GeLU_tanh")[:, None] & (v == "GeLU")[None, :]))
                if not _p5("gelu"):             # (pass 4: erf and tanh GeLU walled from each other)
                    m &= ~gelu_cells
                    no_up = no_up | gelu_cells
                if lever and _p5("act_width"):  # activation width: k host units per target unit, FFN x k
                    fb = flag("ffn_bias").astype(np.int64)
                    ua = sorted(set(v.tolist()))
                    ai = np.array([ua.index(x) for x in v])
                    K = np.array([[[act_width_k(a, b, bool(c)) for c in (0, 1)] for b in ua]
                                  for a in ua], dtype=np.int64)
                    kk = K[ai[:, None], ai[None, :], fb[None, :]]
                    wfit = ~m & ~no_up & (kk > 0)
                    m |= wfit
                    kmul = np.where(wfit, np.maximum(kk * _cr("act_k"), 1), kmul)
            ok &= _why(f"slot:{slot}", m)
        fts = np.array([p["slots"].get("feats", "") for p in P])          # FEATURES no slot carries
        # the eps lever switches a lever-capable host's sandwich post-norms / v-norm off (payable)
        offs = [frozenset(EPS_LEVER_FEATS & set(f.split("+"))) if e and _p5("eps_feats") else frozenset()
                for f, e in zip(fts, eps_h)]
        offs = [o | _fr_host_off(f, sc, lever) for o, f, sc in zip(offs, fts, flag("sc_norm_off"))]
        hk = list(zip(fts.tolist(), offs))
        uf = sorted(set(fts.tolist()))
        uh = sorted(set(hk), key=lambda x: (x[0], sorted(x[1])))
        # fr-positions (fp_nope_layer_host): a RoPE + NoPE host plays a target WITHOUT NoPE layers on its RoPE layers
        # alone, its NoPE layers zeroed (Lean nope_mix_host_skip: identity blocks drop out of the stack, whatever they
        # compute -- Beam's code-fixed NoPE-layer temperature q_temp_nope included); the rotating-layer Hall below
        # charges the layers. A target with NoPE layers of its own keeps both walls (its NoPE layers may land on the
        # host's NoPE layers, where the temperature runs)
        nope_off = lambda a, off: (off | (NOPE_HOST_OFF if (_r3("fp_nope_layer_host")
                                                           and "nope_layers" not in a.split("+")) else frozenset()))
        fm = np.array([[feats_ok(a, b, strict, mode == "mixgap", nope_off(a, off)) for (b, off) in uh] for a in uf])
        fi = np.array([uf.index(x) for x in fts])
        hi = np.array([uh.index(x) for x in hk])
        ok &= _why(f"feature", fm[np.ix_(fi, hi)])
        for name, m in _r3_masks(P, fts, eps_h, hidden).items():     # round 3, batch 1 (classification dims)
            ok &= _why(name, m)
        mixer, window, cla = sa("mixer"), sa("window"), sa("crosslayer_kv")
        plain = (mixer == "softmax") & (cla == "no")      # a plain softmax stack (any window)
        # non-plain targets: the categorical mixer / window / cla orders
        no_gap = strict or mode == "nomix" or (mode == "payable" and not MIXER_GAP_PAYABLE)
        cla_strict = strict or (mode in ("payable", "nomix") and not CLA_CARRY_PAYABLE)
        cat = cat_ok(mixer, "mixer", no_gap) & cat_ok(cla, "crosslayer_kv", cla_strict)
        if mode == "mixgap":                  # the near-gap granted: any host with softmax layers plays recurrent
            has_soft = np.array([any(b in ("softmax", "hybrid") for b in branches(v)) for v in mixer])
            # (round 2, pass 5, transitivity: a KV-sharing host's own-KV layers play a non-sharing recurrent target's
            # attention as they play a plain one's -- the cla ZERO_EDGE, kind_match counting the layers)
            cla_zero = (cla == "no")[:, None] & np.array(["yes" in branches(v) for v in cla])[None, :]
            cat |= has_soft[None, :] & np.array([m != "softmax" for m in mixer])[:, None] & (cat_ok(
                cla, "crosslayer_kv", cla_strict) | cla_zero)    # layers (kind_match charges which layers)
        # (the window slot is no longer a categorical check for non-plain targets: kind_match matches their
        # full / sliding layers one by one, which the model-level label only approximated)
        # plain targets: the ZERO_EDGES. A plain host runs them on all its layers, except a
        # sliding-window host for a full-window target (only its full layers); any other host
        # (hybrid / SSM / KV-sharing) on its own-KV attention layers, full ones for a full target.
        # With the context axis on, a full target is reached by depth (win_factor) instead.
        # Round 2, pass 5, the WINDOW LEVER: a host whose window is config (win_free) sets it >= the context, so its
        # sliding layers are full ones (its full-layer count is its attention-layer count).
        host_full = (flag("win_free") & _p5("window")) | (not _cr("window"))
        # TRANSITIVITY (forced): a window-free TARGET stands for its family's config class, so a host whose window is
        # fixed plays it as its full-window self (X full <= Y window-free <= Z fixed-window must give X <= Z)
        tgt_full = host_full[:, None] & ~host_full[None, :]
        need_full = ((window == "full")[:, None] | tgt_full) & (not USE_CONTEXT)
        h_local, h_plain = ((window == "local") & ~host_full)[None, :], plain[None, :]
        fe = np.where(host_full, dim("attn_eff", "eff_layers"), dim("full_eff", "eff_layers"))
        fp = np.where(host_full, dim("attn_param", "param_layers"), dim("full_param", "param_layers"))
        ae, ap = dim("attn_eff", "eff_layers"), dim("attn_param", "param_layers")
        all_layers = h_plain & ~(need_full & h_local)
        zero_e = np.where(all_layers, el[None, :], np.where(need_full, fe[None, :], ae[None, :]))
        zero_p = np.where(all_layers, pl[None, :], np.where(need_full, fp[None, :], ap[None, :]))
        plain_t = np.broadcast_to(plain[:, None], (n, n))
        if _r3("d_transitivity_mixer_cla") and not cla_strict:
            # round 3 (d_transitivity_mixer_cla): where the CLA carry is granted (mixgap), a KV-sharing softmax target
            # is a plain one after the carry, so on a host its categorical orders refuse (a hybrid host: no softmax ->
            # hybrid order) it reaches the host by the ZERO_EDGES as a plain target does (carry, then zero the recurrent
            # / sharing layers: the chain A <- plain <- hybrid composed), charged the host's own-KV attention layers
            # (zero_e). Where the categorical path holds it keeps it (the host's whole depth, never less than zero_e).
            # Target side only: h_plain stays literal
            carry = (mixer == "softmax") & (cla == "yes")
            plain_t = plain[:, None] | (carry[:, None] & ~cat)
        ok &= _why(f"mixer/cla", plain_t | cat)
        # the rot gate: a RoPE-family target on a RoPE-family host needs the host's inv_freq tables to
        # hold the target's rotating pairs and NoPE slots (hf/partial_rope_host.py, rope_fits). Round 2, pass 5 (A3):
        # only a pairs-free host (rope_pairs_free: longrope) sets its buffer freely; a fixed-rotary host sets rope_theta
        # (and a fraction-free one its rotary fraction) only -- fixed_rot_fits; a fraction-free host's tables reach its
        # whole head (host_rope_cap)
        gated = rot_t[:, None] & rot_h[None, :]
        if gated.any():
            need = [tuple(p["dims"].get("rope_sched", ())) for p in P]
            cap = [host_rope_cap(p) for p in P]
            kind = [p["dims"].get("rope_kind", "") for p in P]
            memo = {}
            g = np.zeros((n, n), bool)
            tk = [(need[i], cap[i], bool(pf[i]), bool(ff[i]), int(hd[i]), kind[i]) for i in range(n)]
            hk = [(cap[j], bool(pf[j]), int(hd[j]), bool(ff[j]), kind[j], need[j]) for j in range(n)]
            ut, uh = {k: x for x, k in enumerate(dict.fromkeys(tk))}, {k: x for x, k in enumerate(dict.fromkeys(hk))}
            ti, hi = np.array([ut[k] for k in tk]), np.array([uh[k] for k in hk])
            gi, gj = np.nonzero(gated)
            comb, first = np.unique(ti[gi] * len(uh) + hi[gj], return_index=True)
            res = {}
            for c, f in zip(comb.tolist(), first.tolist()):
                i, j = int(gi[f]), int(gj[f])
                key = c
                if key not in memo:
                    # TRANSITIVITY (forced, round 2 pass 5): a target is its family's config class as a host. A
                    # fraction-free target on a host that is not plays as its whole head rotating; a fixed-rotary host
                    # plays only fixed-rotary targets (a pairs-free target hosts any frequency schedule, which a fixed
                    # host cannot form), and a fraction-free one only on a fraction-free host
                    # Round 3 (d_transitivity_rot_gate): a FIXED target whose tables drive more than its schedule
                    # (cap != sched: a "proportional" / zero-tail table, as a host it rotates up to cap) stands for
                    # its cap too -- else A <= B (A on B's cap) and B <= C (B's sched on C) without A <= C (Z3
                    # counterexample triple); on a fixed host it is matched as its cap, and a
                    # fixed host of that kind plays its own shape as its cap (reflexive)
                    rg = _r3("d_transitivity_rot_gate")
                    big = lambda x: rg and not pf[x] and not ff[x] and tuple(cap[x]) != tuple(need[x])
                    nd = cap[i] if (pf[i] or ff[i] or big(i)) else need[i]   # the target's class: what its tables reach
                    if not _p5("rot_config"):       # (pass 4: any RoPE host sets its buffer)
                        memo[key] = rope_fits(need[i], cap[j])
                    elif pf[j]:
                        memo[key] = rope_fits(nd, cap[j])
                    else:
                        memo[key] = (not pf[i] and (ff[j] or not ff[i])
                                     and fixed_rot_fits(cap[i] if big(i) else need[i], int(hd[i]), kind[i], cap[j],
                                                        int(hd[j]), bool(ff[j]), kind[j],
                                                        cap[j] if big(j) else need[j]))
                res[c] = memo[key]
            lut = np.zeros(len(ut) * len(uh), bool)
            for c, val in res.items():
                lut[c] = val
            g[gi, gj] = lut[ti[gi] * len(uh) + hi[gj]]
            # round 3 (nope_rope_limit, a host config lever, a LIMIT): a target's NoPE layers on a host table of their
            # own scaled to infinity (rope_lin 2: one linear factor per table), its rotating schedules on the host's
            # other tables (nope_rope_limit_rot) -- payable / mixgap only
            mixed = np.array([any(r < 0 for r in x) and any(r > 0 for r in x) for x in need])
            lv = gated & ~g & mixed[:, None] & (lin_h & (dim("rope_lin") >= 2))[None, :]
            for i, j in zip(*np.nonzero(lv)):
                key = ("lin", ti[i], hi[j])
                if key not in memo:
                    memo[key] = nope_rope_limit_rot(need[i], int(hd[i]), kind[i], bool(pf[i]), bool(ff[i]), cap[j],
                                                    int(hd[j]), bool(pf[j]), bool(ff[j]), kind[j], 2)
                g[i, j] = memo[key]
            ok &= _why(f"pos rot gate", ~gated | g)
            # KV-head packing lets a rope_pairs_free host B (longrope: repeated frequencies) play any RoPE target X
            # with qheads_X <= qheads_B, even with more KV heads than B has (X MHA, head_dim hd_B / g_B, packed g_B
            # to a head). For transitivity (X <= B <= C must give X <= C) a fixed-rotary host C, which cannot pack
            # RoPE heads, plays such a TARGET B only as that MHA expansion: kv_heads_C >= qheads_B (a conservative
            # wall for fixed-rotary GQA hosts -- InternLM2, MiniCPM, DeciLM, Qwen-1, ... -- of Llama-style targets)
            ok &= _why(f"pos fixed-rotary packing", ~gated | ~(pf[:, None] & ~pf[None, :]) | (dim("kv_heads")[None, :] >= qh[:, None]))
        usable_e = np.where(plain_t, zero_e, el[None, :])
        usable_p = np.where(plain_t, zero_p, pl[None, :])
    if mode != "capacity":
        # K = V: the target's independent-V layers need the host's independent-V capacity
        ok &= _why(f"K=V", ((dim("kv_cap_u", "kv_cap")[None, :] >= dim("kv_cap_u", "kv_cap")[:, None])
               & (dim("q_cap_u", "q_cap")[None, :] >= dim("q_cap_u", "q_cap")[:, None])
               & (dim("qheads_u", "qheads")[None, :] >= dim("qheads_u", "qheads")[:, None])
               & (dim("head_dim_u", "head_dim")[None, :] >= dim("head_dim_u", "head_dim")[:, None] + add_hd)))
    if mode != "capacity" and not MLA_FOLD_PAYABLE:       # MLA <-> plain attention: a wall both ways (KV_ORDER)
        mla = np.array(["MLA" in branches(v) for v in kvv])                 # (a host may carry both: 'MHA|MLA')
        plain_kv = np.array([any(b != "MLA" for b in branches(v)) for v in kvv])
        ql, kl, qs = dim("q_latent"), dim("kv_latent"), dim("mla_qswitch")
        # round 2, pass 5: a plain target on an MLA-only host under the eps lever (the latent norms switched off,
        # payable): the shared rotary key carries the target's rotating keys only for ONE KV head (MQA-RoPE), or for
        # several packed side by side on a pairs-free host (rope_fits on kv_heads x the target's dims); a NoPE target
        # needs none. Its keys' non-rotating dims and its values come from the latent (qk_nope, v_head_dim per head,
        # the latent rank min(hidden, kv_heads (2 hd - d_rot)), the query latent min(hidden, q_cap) unless switchable)
        posv = sa("pos")
        pm = (~mla)[:, None] & (mla & ~plain_kv)[None, :] & eps_h[None, :] & _p5("eps_mla_plain")
        if pm.any():
            kvh = dim("kv_heads")
            rot_d = np.array([max([r for r in p["dims"].get("rope_sched", ()) if r > 0], default=0)
                              if p["slots"]["pos"] in ROT_TARGETS else 0 for p in P], dtype=np.int64)
            shape_ok = np.zeros((n, n), bool)
            for i, j in zip(*np.nonzero(pm)):
                if posv[i] == "nope":
                    shape_ok[i, j] = True
                elif posv[i] in ROT_TARGETS:
                    sched = tuple(r for r in P[i]["dims"].get("rope_sched", ()) if r != 0)
                    k = int(kvh[i]) or 1
                    shape_ok[i, j] = ((k == 1 or bool(P[j]["dims"].get("rope_pairs_free")))
                                      and rope_fits(tuple(k * r for r in sched), (int(P[j]["dims"].get("mla_rope", 0)),)))
            need_kl = np.minimum(hidden, kvh * (2 * hd - rot_d))
            pm &= (shape_ok & (kl[None, :] >= need_kl[:, None])
                   & (dim("mla_nope")[None, :] >= (hd - rot_d)[:, None]) & (dim("mla_v")[None, :] >= hd[:, None])
                   & ((ql[None, :] == 0) | (qs[None, :] > 0) | (ql[None, :] >= np.minimum(hidden, q)[:, None])))
        ok &= _why(f"kv MLA<->plain", np.where(mla[:, None], mla[None, :], plain_kv[None, :] | pm))
        # (the shared rotary key: the rot gate above, rope_fits on the MLA's one table of qk_rope_head_dim dims)
        # round 2, pass 3: the latent norms (q_a_layernorm / kv_a_layernorm) are built with a HARD-CODED eps 1e-6 in
        # every MLA port (not rms_norm_eps), so a zero-padded wider latent is exact only up to eps. A host latent that
        # is an integer MULTIPLE m of the target's stores the latent m times (kv_a_proj rows repeated, kv_b / q_b read
        # the first copy): the rms over m copies is the rms of one, and the norm is reproduced exactly with the same
        # eps. Otherwise a wall in the exact slice; round 2, pass 5: under the eps lever (the latent input scaled by
        # sqrt(kl_h / kl_t), eps_eff = the target's) any wider latent on a lever-capable host (payable). The universal
        # host's latent norm averages over the target's block with the target's eps (dims mla_lat_free).
        # Divisibility composes, so the rule stays transitive. The per-head widths are compared on
        # their own: the latent-sourced nope part of q / k (mla_nope) and v_head_dim (mla_v), host >= target (zero
        # pad; the softmax scale folds into q_b_proj) -- head_dim = nope + rope let a host with a wider shared rotary
        # key stand in for per-head nope dims.
        mult = lambda h, t: (t[:, None] == 0) | ((h[None, :] >= t[:, None]) & (h[None, :] % np.maximum(t[:, None], 1) == 0))
        lat_free = ((dim("mla_lat_free") > 0)[None, :] | (eps_h[None, :] & _p5("eps_latent"))
                    | (not _cr("mla_mult")))
        ok &= _why(f"kv MLA latents", ~(mla[:, None] & mla[None, :]) | ((((ql[None, :] > 0) == (ql[:, None] > 0)) | (qs[None, :] > 0))
                                                 & (ql[None, :] >= ql[:, None]) & (kl[None, :] >= kl[:, None])
                                                 & ((mult(kl, kl) & (mult(ql, ql) | (qs[None, :] > 0))) | lat_free)
                                                 & (dim("mla_nope")[None, :] >= dim("mla_nope")[:, None])
                                                 & (dim("mla_v")[None, :] >= dim("mla_v")[:, None])))
    # TransMLA penalty (the capacity view): EVERY host without an MLA branch materialises MLA's K and V and
    # pays the same factor -- GQA / MQA, MHA and the no-KV ('none') recurrent stacks alike; charging only
    # GQA / MQA made MLA <- MHA / none <- GQA hold while MLA <- GQA failed (non-transitive, even in capacity)
    mla_t = np.array(["MLA" in branches(v) for v in kvv])
    penalty = np.where(mla_t[:, None] & ~mla_t[None, :], MLA_PENALTY if _cr("mla_penalty") else 1, 1)
    # the sink's spare head dims widen the target's heads (q_cap, kv_cap scale with the head dim)
    hd_s = np.maximum(hd, 1)[:, None]
    kv_t = np.where(add_hd > 0, kv[:, None] * (hd[:, None] + add_hd) // hd_s, kv[:, None])
    q_t = np.where(add_hd > 0, q[:, None] * (hd[:, None] + add_hd) // hd_s, q[:, None])
    ok &= _why(f"kv_cap", kv[None, :] >= kv_t * penalty)
    ok &= _why(f"q_cap", q[None, :] >= q_t)
    if _cr("kv_heads"):
        ok &= _why(f"kv heads", kv_heads_fit(P, mode, add_hd if add_hd.any() else None))  # KV heads by count, with KV-head packing
    ok &= _why(f"qheads", qh[None, :] >= qh[:, None])  # query-head COUNT (richness)
    ok &= _why(f"head_dim", hd[None, :] >= hd[:, None] + add_hd)  # Q/K head_dim: a narrow-head model cannot
    #                                                        reproduce a wide head's attention pattern
    #                                                        (score rank <= head_dim; V budget stays in kv_cap)
    block = sa("block")
    par_emu_seq = (block[None, :] == "parallel") & (block[:, None] == "sequential")  # B par emulates A seq
    seq_emu_par = (block[None, :] == "sequential") & (block[:, None] == "parallel")  # B seq emulates A par
    loop_v = sa("loop")
    par_comp = np.zeros((n, n), bool)
    if mode != "capacity":
        # round 2, pass 5: parallel <= seq by the 1-D RMS compensator (payable, measured families: rms_ok): the
        # vanishing carry of the block input rescaled by k units per carried channel, k d units per layer, a pad
        # (round 3, batch 3b, spot_comp_removed: off -- no host construction builds it (verify/prenorm_rescue/ measures it), see the norm slot above)
        if not strict and mode != "mixgap" and not PAR_CARRY_PAYABLE and p5_port_live("comp_par"):
            kc = np.where(frank >= FFN_RANK["glu"], COMPENSATOR_K["glu"],
                          np.where(flag("ffn_bias") & np.isin(sa("act"), list(BOS_ODD_LINEAR_ACTS)),
                                   COMPENSATOR_K["relu"], 0))
            par_comp = seq_emu_par & flag("rms_ok")[:, None] & (kc > 0)[None, :]
            add_h += np.where(par_comp, 1, 0)
            f_add += np.where(par_comp, kc[None, :] * hidden[:, None], 0)
        # round 2, pass 5: an untied target on a looped host -- payable only on a plain looped host (Nanbeige4.2: the
        # counter + start-token marker, measured), +2 channels and 2d + 2 units (+2 / layer if it norms between loops,
        # +4d / layer with a v bias); every other looped host is a wall for it
        lp = (loop_v == "untied")[:, None] & (loop_v == "looped")[None, :]
        lp_ok = lp & flag("loop_plain")[None, :] & (not strict) & _p5("loop")
        ok &= _why(f"loop", ~lp | lp_ok | (not _p5("loop")))
        add_h += np.where(lp_ok, LOOP_EXTRA_CHANNELS, 0) * _cr("loop")
        f_add += np.where(lp_ok, 2 * hidden[:, None] + 2 + 2 * dim("loop_norm")[None, :]
                          + 4 * hidden[:, None] * dim("loop_vbias")[None, :], 0) * _cr("loop")
    if _r3("w1_par_units"):
        # round 3, batch 3b (verify/r3_notes.md W1-1): a sequential host of a parallel target (Lean parallel_sub_seq_2x,
        # norm-free) needs 2 d width AND 2 d_target extra MLP units per block -- the identity pairs phi(v) - phi(-v) = v
        # moving the attention write out of the spare channel: ffn_target + 2 d_target units, wherever the 2x width
        # applies (mode capacity, and mixgap's granted carry; payable / exact wall the pair under a pre-norm)
        f_add += np.where(seq_emu_par, 2 * hidden[:, None], 0) * _cr("par_width")
    ok &= _why(f"ffn width", (fT[None, :] >= np.maximum(kmul * fT[:, None] + f_add, f_min)) & (
        fA[None, :] >= np.maximum(kmul * fA[:, None] + f_add, f_min)))  # FFN capacity (x k, + units, a floor)
    # per-layer FFN widths (Nemotron-NAS blocks, OpenELM): each target layer needs its own host layer at least as
    # wide -- the sorted widths dominate (_ffn_widths_fit); only pairs with a listed model differ from the line above
    fw = [tuple(p["dims"].get("ffn_widths") or ()) for p in P]
    listed = [i for i, w in enumerate(fw) if w]
    if listed:
        lay = lambda i: fw[i] or [int(fT[i]), int(el[i])]           # (a list = one width on every layer)

        def scaled(i, j):                                            # the target's widths x k + units (pass 5)
            k, a = int(kmul[i, j]), int(f_add[i, j])
            if k == 1 and not a:
                return lay(i)
            return tuple(k * x + a for x in fw[i]) if fw[i] else [k * int(fT[i]) + a, int(el[i])]
        before = ok.copy() if explain is not None else None
        for i in listed:
            for j in range(n):
                if ok[i, j] and i != j:
                    ok[i, j] = _ffn_widths_fit(scaled(i, j), lay(j))
                if ok[j, i] and i != j:
                    ok[j, i] = _ffn_widths_fit(scaled(j, i), lay(i))
        if explain is not None:
            _why("ffn per-layer widths", ~(before & ~ok))
    depth2 = ((frank[None, :] < frank[:, None]) | par_emu_seq) & bool(_cr("depth2"))   # 2x depth
    ok &= _why(f"depth (eff)", usable_e >= np.where(depth2, MOE_DEPTH_FACTOR * el[:, None], el[:, None]) + add_l)  # unrolled depth
    ok &= _why(f"depth (param)", usable_p >= pl[:, None] + add_l)  # weight-distinct layers
    if mode != "capacity":
        # seq host for a parallel target: the 2x-width carry of the block input is exact only norm-free (and
        # costs 2 d width plus 2 d MLP units per block there: Lean parallel_sub_seq_2x, the units charged above under
        # w1_par_units; round 3, batch 3b: the old "(L+1) d, not 2d: parallel_sub_seq_wide" was wrong); under a stock
        # pre-norm it fails (~1e-2, block/prenorm_carry.py; parallel_sub_seq_prenorm_fails) -- the CLA carry's
        # problem, not payable (round 2, pass 5: payable by the compensator for the measured families, par_comp
        # above; round 3, batch 3b: that credit is removed, spot_comp_removed)
        if not PAR_CARRY_PAYABLE and mode != "mixgap":
            ok &= _why(f"block parallel<-sequential", ~seq_emu_par | par_comp)
        # dense host for a MoE target: exact (a hard-select / beta -> inf limit, Lean dense_hard_select) only for
        # a TOP-1 router with a unit gate (renormalised); top-k > 1 or a gate times the router score p(x)
        # (Switch, Llama-4, norm_topk_prob = False) adds a multiplicative factor no construction prices
        dense_h = (frank <= FFN_RANK["glu"])[None, :] & (frank >= FFN_RANK["moe"])[:, None]
        # round 2, pass 3: payable, not exact -- the pre-norm select / mask construction is measured (moe/
        # dense_host_prenorm.py, branch moe-loop-rescue: top-1 100% at its sharp constants) and Lean has dense ⊇ MoE
        # only as a limit (continuous_host_ne_top1_moe), so the exact slice has no dense host for a MoE target
        ok &= _why(f"ffn MoE<-dense", ~dense_h | ((dim("moe_unit_top1") > 0)[:, None] & (not strict)))
        # round 3, batch 3b (spot_glu_vanilla, verify/spot/SPOT.md): a gated target on a VANILLA (ungated) FFN host has
        # no construction: one vanilla unit act(w.x) cannot form act(g) u, and the 2x-depth route (Lean
        # swiglu_eq_sq_ffn: a squared-unit FFN of act(g) and u, or the yarotsky_product_bound for ReLU) needs act(g)
        # carried across the next pre-norm, which rescales it by the token's rms (channel_carry_prenorm_fails,
        # block/prenorm_carry.py). The same for a MoE target on a vanilla host: the dense-host route
        # (moe/dense_host_prenorm.py) builds its select masks t^2 sigma(beta t) from SwiGLU units and its experts as
        # masked GLU units. A wall (needs theorem glu_sub_vanilla_prenorm); the 2x depth stays for the dense GLU host
        # of a top-1 MoE target (moe_two_layer_margin, measured) and the parallel host of a sequential target
        if _r3("spot_glu_vanilla"):
            # (round 3, mamba_ffn_none: an FFN-less target is no gated one -- any FFN host zeroes its MLP)
            ok &= _why("ffn gated<-vanilla", ~((~np.isin(ffn, ("vanilla", "none")))[:, None] & (ffn == "vanilla")[None, :]))
        # round 3, batch 3b (spot_vanilla_glu_bias, catalog p5_vanilla_on_glu_nobias): a vanilla target on a gated /
        # MoE host holds the up branch at 1 (Lean glu_ungated: up weights 0, BIAS 1); with no up bias the only constant
        # is a residual channel, which the pre-norm divides by the token's rms (top-1 76%): the host needs bias_up
        # TRANSITIVITY (forced): a target with an up bias stands for its class -- it hosts the vanilla targets through
        # that bias -- so it needs a host with one too; and its own up bias is a constant no bias-free host forms (the
        # bias_qkv argument: a constant channel is rescaled by the pre-norm). Without this, vanilla X <= biased GLU Y
        # <= bias-free Z gave X <= Y <= Z without X <= Z (mamba / mpt-test <- basal / hubble <- Phi-3.5-MoE / TW3)
        if _r3("spot_vanilla_glu_bias"):
            bu = flag("bias_up")
            ok &= _why("ffn vanilla<-gated up bias", ~((ffn == "vanilla")[:, None] & (ffn != "vanilla")[None, :]
                                                      & ~bu[None, :]) & ~(bu[:, None] & ~bu[None, :]))
        # round 3, batch 3b (hadamard_mlp_unlisted): a HadamardMLP is hosted only by its own code at equal n (n fixes
        # the Kronecker blocks and the seeded permutations), and hosts nothing else (its maps are Kronecker-structured,
        # not dense d x inter matrices; needs theorems: a plain FFN without an input-dependent pre-activation scale is
        # not silu(cond(x) M x + b), and a Kronecker n x n map does not span dense maps)
        if _r3("hadamard_mlp_unlisted"):
            hn = dim("hada_n")
            ok &= _why("ffn hadamard", hn[:, None] == hn[None, :])
        # ALiBi <- (partial) RoPE: the frozen small-angle pair is odd in i - j (causal targets only) and needs a
        # spare head-dim pair at inv_freq = 0 besides the target's dims (Lean alibi_sub_rope_*, alibi_pair_odd);
        # round 2, pass 5: set by the host's config only (the pos slot above: pairs-free, or zero_freq_room)
        posv = sa("pos")
        al = (posv == "alibi")[:, None] & np.isin(posv, ("rope", "partial_rope"))[None, :]
        ok &= _why(f"pos alibi<-rope", ~al | ((sa("gen") == "ar")[:, None] & (hd[None, :] >= hd[:, None] + 2)))
        # ALiBi <- ALiBi: the host's code-computed slopes must include the target's (alibi_nests: BLOOM / MPT /
        # Baichuan share one slope set per head count, Falcon's are bf16-rounded); NoPE <- ALiBi only where a config
        # value removes the bias (alibi_off: Falcon's alibi flag, the remote MPT's alibi_bias_max). For
        # transitivity (NoPE <= switchable X <= Y must give NoPE <= Y) a switchable ALiBi target needs a
        # switchable ALiBi host too: a conservative wall for BLOOM / Baichuan / native-MPT hosts of remote-MPT /
        # Falcon targets (they play them only through the slopes they cannot switch off).
        aa = (posv == "alibi")[:, None] & (posv == "alibi")[None, :]
        if aa.any():
            # (round 3, falcon_alibi_slope_scale: the slopes' scale, 1 / sqrt(head_dim) for Falcon, 1 elsewhere)
            sig = [(int(p["dims"].get("alibi_heads", 0)), int(p["dims"].get("alibi_max", 8) or 8),
                    bool(p["dims"].get("alibi_bf16", 0)), float(p["dims"].get("alibi_scale", 1.0) or 1.0)) for p in P]
            us = sorted(set(sig))
            nm = np.array([[alibi_nests(a, b) for b in us] for a in us])
            si = np.array([us.index(x) for x in sig])
            off = dim("alibi_off") > 0
            ok &= _why(f"pos alibi<-alibi", ~aa | (nm[np.ix_(si, si)] & (~off[:, None] | off[None, :])))
        na = (posv == "nope")[:, None] & (posv == "alibi")[None, :]
        ok &= _why(f"pos nope<-alibi", ~na | (dim("alibi_off") > 0)[None, :])
    if mode != "capacity":
        # recurrent capacity per family (round 2, pass 4); round 2, pass 5: Mamba-2 groups under the eps lever
        # (payable). mode="mixgap" grants attention-for-recurrent, so there it binds only on a host with no
        # attention layer to stand in (verify/REVIEW.md finding 8: a d_state-16 Mamba hosted a d_state-32 one)
        rec = _rec_ok(P, lever=not strict and _p5("eps_rec")) | (not _cr("rec"))
        if mode == "mixgap":
            attn_h = sum(dim(f"k{k}_eff") for k in ("FU", "FT", "LU", "LT")) > 0
            attn_h |= np.array([any(b in ("softmax", "hybrid") for b in branches(p["slots"].get("mixer", "softmax")))
                                and not any(f"k{k}_eff" in p["dims"] for k in LAYER_KINDS) for p in P])
            rec = rec | attn_h[None, :]
        ok &= _why(f"recurrent dims", rec)
    if mode != "capacity":
        ok &= _why(f"ffn MoE routing", _moe_ok(P, ok, hidden, fT, kmul, strict))  # MoE routing, dense / MoE layer kinds, dense host for a MoE (moe_why)
    if mode != "capacity":                     # every target layer needs a host layer of a kind that reaches it
        loose = mode == "mixgap"               # the near-gap (attention plays recurrent) and the CLA carry free
        reach = ({h: t + (("R", "S") if h in ("FU", "FT", "LU", "LT") else ()) for h, t in KIND_REACH.items()}
                 if loose else dict(KIND_REACH))
        if USE_CONTEXT:                        # the context axis prices window reach in depth (win_factor):
            reach["LU"] = reach["LU"] + ("FU", "FT")          # a sliding layer may stand for a full one
            reach["LT"] = reach["LT"] + ("FT",)
        if reach == KIND_REACH:
            reach = None
        af = add_l if add_l.any() else None
        ok &= _why(f"layer kinds", (not _cr("layer_kinds")) | (kind_match(P, "eff", np.where(depth2, MOE_DEPTH_FACTOR, 1), reach, host_full, af)
               & kind_match(P, "param", 1, reach, host_full, af)))
        # fr-positions (fp_nope_layer_host): a target's ROTATING layers need ROTATING host layers -- the same Hall
        # matching on the rk* counts (a RoPE + NoPE mix offers / demands its RoPE layers only; every other point all of
        # its attention layers). One count vector serves demand and supply, so it is transitive as kind_match is; with
        # no mix in P it is kind_match itself
        Pr = _rot_points(P) if _r3("fp_nope_layer_host") else None
        if Pr is not None:
            ok &= _why(f"pos rotating layers", (not _cr("layer_kinds")) |
                       ((kind_match(Pr, "eff", np.where(depth2, MOE_DEPTH_FACTOR, 1), reach, host_full, af)
                        & kind_match(Pr, "param", 1, reach, host_full, af))))
    width = np.where(seq_emu_par, 2 if _cr("par_width") else 1, 1)        # seq hosts parallel: 2x width
    sub_add = 0
    if mode != "capacity":                     # LayerNorm host for an RMSNorm target: the [h; -h] mirror
        norm = sa("norm")
        mir = (norm[:, None] == "RMSNorm") & (norm[None, :] == "LayerNorm")
        if _r3("cost_norm_subspace"):
            # branch fr-cost (hf/norm_subspace.py, Lean NormSubspace.rmsNorm_sub_layerNorm_subspace): the host residual
            # U h, U a d_h x d isometry onto the mean-zero subspace (exists iff d_h >= d + 1), is a no-op for the
            # centring and keeps |h|: hidden + 1, not 2x. Not where a gained norm output is written INTO the residual
            # (NORM_SUBSPACE_DENY on either side, an embedding norm: the gain must commute with U, i.e. the mirror),
            # nor on a tied host (the final norm's gain cannot fold into lm_head = E^T), nor with the parallel carry
            sub = mir & ~seq_emu_par & _norm_subspace_ok(P)[:, None] & _norm_subspace_ok(P, host=True)[None, :]
            mir = mir & ~sub
            sub_add = np.where(sub, 1, 0)
        width = width * np.where(mir, NORM_WIDTH_FACTOR if _cr("norm_mirror") else 1, 1)
    ok &= _why(f"hidden", hidden[None, :] >= width * hidden[:, None] + add_h + sub_add)
    if USE_CONTEXT:                                   # max-context DEMAND + tight window<->depth<->context
        # round 3 (max_context_unknown): the context codes (_ctx_codes: read / CTX_UNBOUNDED / CTX_UNKNOWN) compared by
        # _ctx_ge -- an unknown host supplies only an unknown target, an unknown target needs an unknown or unbounded
        # host, an unbounded host supplies anything (unknown is its own value: reflexive, and still a preorder)
        cu = _r3("max_context_unknown")
        mc, ewin = ((_ctx_codes(P, "max_context"), _ctx_codes(P, "eff_window")) if cu
                    else (dim("max_context"), dim("eff_window")))
        # round 3, batch 2b (design decision 3: as for DSA): a context-gated point (dims pos_gate: Qwen-1
        # logn / dynamic NTK, the Llama-4-style temperatures, Meditor's NTK) is plain attention only below its gate, so
        # an edge into it is credited only up to the gate -- the caveat: nothing is claimed past it. The cap applies to
        # the host's supply too (a host whose gate is config, Qwen's seq_length, could lift it; not used: the same cap
        # on both sides keeps the context order a total preorder, so transitive)
        if _r3("position_query_temperature") or _r3("dynamic_ntk_in_context"):
            pg = dim("pos_gate")
            mc = np.where(pg > 0, np.minimum(mc, pg), mc)     # (an unknown code stays CTX_UNKNOWN: min(-1, g))
        if cu:
            ok &= _why(f"context", _ctx_ge(mc[None, :], mc[:, None]))
        else:
            ok &= _why(f"context", mc[None, :] >= mc[:, None])  # host must support the target's context length (hard)
        # a local host reaches a wider target only by ALSO paying depth ceil((reach-1)/(w-1)) --
        # window and depth spent together, the tight case a one-way order can't express:
        w_ge = _ctx_ge(ewin[None, :], ewin[:, None]) if cu else ewin[None, :] >= ewin[:, None]
        win_factor = np.where(w_ge, 1.0, np.ceil((ewin[:, None] - 1) / np.maximum(ewin[None, :] - 1, 1)))
        w_ok = usable_e >= win_factor * el[:, None]
        if cu:   # (round 3: an unknown reach on either side, not supplied by _ctx_ge, has no depth price: not credited)
            w_ok &= w_ge | ((ewin[None, :] >= 0) & (ewin[:, None] >= 0))
        ok &= _why(f"context depth", w_ok)
    if USE_VOCAB:
        vocab = dim("vocab")
        ok &= _why(f"vocab", vocab[None, :] >= vocab[:, None])  # embedding/head capacity (opt-in)
    aw_drop = None
    if _r3("spot_act_width_oai") and mode != "capacity" and (kmul > 1).any():
        # TRANSITIVITY (forced, round 3 batch 3b): with the act-width fits off on a swigluoai host (act_width_k), a
        # target Y credited on a swigluoai host Z through its own activation's cell (SiLU <- swigluoai: the 1.702 fold
        # inside the clamp, ACT_LIMIT) that natively hosts a target X through an act-width fit (X GeLU on Y SiLU at
        # FFN x k) would give X <= Y <= Z without X <= Z: as for the compensator, Y keeps the swigluoai host only if it
        # hosts no act-width target in the set (hubble-1b <- gpt-oss, which hubble plays mpt-test GeLU through)
        act_a = sa("act")
        aw_host = (ok & (kmul > 1)).any(0)
        aw_drop = aw_host[:, None] & (act_a == "swigluoai")[None, :] & (act_a != "swigluoai")[:, None]
    if (ln_comp.any() or par_comp.any()) and mode != "capacity":
        # TRANSITIVITY (forced, round 2 pass 5): the compensator credits a target only within the measured RMS
        # range, a property of the function it computes; a credited target Y that natively hosts an UNMEASURED
        # target X of the same kind (a parallel X for parallel <= seq, a LayerNorm-beta X for the fold) would give
        # X <= Y <= Z without X <= Z, so Y keeps the credit only if every such X it hosts is measured too
        oob = ~flag("rms_ok")
        par_t = sa("block") == "parallel"
        lnb_t = (sa("norm") == "LayerNorm") & flag("ln_beta")
        bad_par = (ok & (par_t & oob)[:, None]).any(0)
        bad_ln = (ok & (lnb_t & oob)[:, None]).any(0)
        drop = (par_comp & bad_par[:, None]) | (ln_comp & bad_ln[:, None])
        ok &= _why("compensator: hosts an unmeasured target", ~drop)
    if aw_drop is not None:
        ok &= _why("swigluoai host: the target hosts an act-width target", ~aw_drop)
    return ok


def _moe_scaled(sig, k):
    """moe_sig with every FFN width x k (the activation-width rule: k host units per target unit)."""
    if k == 1 or not sig:
        return sig
    T = _moe_fields(sig)
    T.update(ew=T["ew"] * k, shared=T["shared"] * k, dense_w=T["dense_w"] * k,
             par_res=T["par_res"] * k)                      # (round 3, batch 3a: Arctic's residual MLP width)
    return tuple(T[f] for f in MOE_SIG)


def _moe_ok(P, ok, hidden, fT, kmul=None, strict=False):
    """[i, j]: moe_why finds nothing against host P[j] playing target P[i]'s FFN routing (round 2, pass 3).

    MoE <- MoE (Lean MoERouting.lean, the duplication family: m router-row copies of each target expert):
      top-k is FIXED config, so the host's k must be a positive multiple m of the target's; E_h >= m E_t; the expert
        width m f_h >= f_t (m slices with the down-projection x m, moeR_dup_renorm); the shared expert compared on
        its own, s_h >= s_t (a routed expert cannot be always on; a shared-less host for a shared target would need a
        second layer, the parallel-carry problem); the expert-input rank (Nemotron-3 moe_latent_size: fc1 / fc2
        latent projections around the experts, linear, so they fold into the experts: rank = latent, else hidden)
        host >= target; routed_scaling_factor folds (moeR_scaling_fold);
      renormalisation (norm_topk_prob) is free config WHERE THE CODE READS IT (RENORM_FIXED / RENORM_KEYS): a
        renormalised target is exact under copies; an unrenormalised SOFTMAX target only with exactly m E_t host
        experts (moeR_dup_plain; extras are only a limit, moeR_dup_plain_extras_ne) and, for transitivity, a host
        whose own router is unrenormalised softmax too;
      extra experts (E_h > m E_t) must score below the copies: a selection bias (-M) or a free hidden dim the
        embedding fills with a constant (read as -M c / rms; its norm folds into the host eps, const_norm_carry);
      the router score (softmax / sigmoid / sqrtsoftplus) is free config only where the code reads it
        (SCORE_CONFIG_TYPES), else the host's code must compute the target's;
      group-limited routing (n_group / topk_group free config): copies in their own groups, m n_group groups keeping
        m topk_group (moeR_group_copies) -- the same group-score code, the host's experts in m n_group equal groups,
        and m = 1 under a top-2-sum group score (moeR_group_top2sum_dup_flips);
      the selection bias is the router_bias feature, Llama-4's input-side gate router_in_gate.
    Round 3, batch 3a (moe_why, MOE_SIG_R3): SparseMixer only at m = 1 (spot_phimoe_dup); a 0-filled group limit and
      Intern-S1-Pro's per-group router only on their own code with the target's experts (extras / copies need theorems
      moeR_group_copies at fill 0 / moeR_per_group_copies; a -inf target on a 0-fill host needs group_zero_fill_shift);
      LongCat's identity slots H = m T; the latent-norm sites (LATENT_NORM_PLAYS); branch post-norms both ways a wall
      across (double_rmsnorm_fold); LLaDA2.2's block routing only on the same block routing at m = 1 (a block host plays
      per-token targets by the expert_capacity lever); an Arctic host's MoE plays no sequential MoE layer (2x depth,
      not plumbed), its residual MLP the dense layers; Meditor-2's two-level router only on its own shape, no dense host.
    TRANSITIVITY: a host can be SET to a renorm / score / group shape its code allows, but as a target it is its
    own config, so config freedom on the host side can break X <= Y <= Z. The one rule the checker forced on the
    snapshot: an unrenormalised softmax target needs a host whose OWN router is unrenormalised softmax (with
    exactly m E_t experts). The capability-superset guards (a target whose family can be set to several renorm /
    score / group values needs a host family that can be set to all of them) are not applied (they would drop real
    edges: verify/r3/specs/d_transitivity_moe.md). Round 3, batch 3a: K2-Horizon-375B's settable score
    (router_score_unread) realises the spec's triple on the snapshot -- small softmax MoE targets <- K2-Horizon (set to
    softmax) <- Ling-1T (sigmoid only), 9 pairs in payable / mixgap -- left to the composed-edge closure (d_transitivity_moe).
    Dense / MoE layer kinds (first_k_dense_replace, mlp_layer_types, moe_layer_freq, ..., _moe_layer_mask): a MoE
      target layer needs a host MoE layer; a dense target layer a host dense layer at least as wide or a spare host
      MoE layer acting dense (dense <= MoE: zero router, the units dealt over the k selected experts, x k, + shared:
      k e_w + s >= the dense width). Counts, not positions: the leading-dense stacks align in order.
    Dense host for a MoE target (moe/dense_host_prenorm.py, graph/rescue_rules.py on branch moe-loop-rescue,
      measured top-1 100% on Huihui-MoE-0.8B-2E and gpt-oss-tinystories forced top-1): only a top-1 unit-gate
      router (renormalised), at 2x depth (MOE_DEPTH_FACTOR) and +E residual channels and +2E(E-1) FFN units; top-k
      >= 2 is a wall (Lean: dense ⊇ MoE is never exact, a limit; the masks alone measured ~77% top-1). E = 1 is a
      dense FFN (dims_of), exact at 1x."""
    import numpy as np
    n = len(P)
    sig = [p["dims"].get("moe_sig") for p in P]
    keys = [(sig[i], int(hidden[i]), int(fT[i])) if sig[i] else None for i in range(n)]
    E = np.array([s[0] if s else 0 for s in sig])
    cand = ok & ((E[:, None] > 0) | (E[None, :] > 0))
    out = np.ones((n, n), bool)
    memo = {}
    for i, j in zip(*np.nonzero(cand)):
        if i == j or keys[i] is None or keys[j] is None:
            continue
        kt = keys[i]
        if kmul is not None and kmul[i, j] != 1:          # the activation-width rule (round 2, pass 5)
            m = int(kmul[i, j])
            kt = (_moe_scaled(kt[0], m), kt[1], kt[2] * m)
        kk = (kt, keys[j])
        if kk not in memo:
            memo[kk] = moe_why(kt, keys[j], strict) is None
        out[i, j] = memo[kk]
    return out


def _point_key_dims():
    """The dims the dedup key must hold: every axis emul_matrix compares UNDER THE CURRENT FLAGS.
    USE_CONTEXT adds max_context and eff_window (the context DEMAND and the window/depth factor),
    USE_VOCAB adds vocab. With both flags off the key is unchanged, so the headline points are too;
    with a flag on, two models that differ only on that axis are no longer merged (a merged class
    would keep only its first member's context / vocab). The ZERO_EDGES layer counts (attn / full,
    unrolled and distinct) and the RoPE schedules (rope_sched / rope_cap, the rot gate) are part of the
    headline relation, so they are always in the key."""
    return (("hidden", "layers", "eff_layers", "param_layers", "kv_cap", "q_cap", "qheads",
             "head_dim", "ffn_total", "ffn_active", "attn_eff", "attn_param", "full_eff", "full_param",
             "rope_sched", "rope_cap",
             "kv_cap_u", "q_cap_u", "qheads_u", "head_dim_u", "kv_latent", "q_latent", "mla_rope",
             "kv_heads", "q_per_kv", "moe_unit_top1", "ffn_widths", "moe_sig", "mla_nope", "mla_v",
             "alibi_heads", "alibi_max", "alibi_bf16", "alibi_off", "alibi_scale", "rope_pairs_free",
             "mla_qswitch", "mla_lat_free", "kv_shapes")      # (REVIEW: compared, now keyed; pass 5 kv_shapes)
            + R3_DIMS                                                                 # round 3, batch 1
            + R3_DIMS_2D                                                              # round 3, batch 2d
            + R3_DIMS_3B                                                              # round 3, batch 3b
            + R3_DIMS_4B                                                              # round 3, batch 4b
            + R3_DIMS_NONLM                                                           # round 3, nonlm_heads
            + R3_DIMS_NOPE                                                            # round 3, nope_rope_limit
            + RC_DIMS + ("rc_rule", "zamba_mem", "zamba_rank", "zamba_attn_adapt")    # round 2, pass 4
            + P5_DIMS                                                                 # round 2, pass 5
            + tuple(f"k{k}_{w}" for k in LAYER_KINDS for w in ("eff", "param"))
            + ("kseq",)                                                               # round 3, kind_order
            + (tuple(f"rk{k}_{w}" for k in ("FU", "FT", "LU", "LT") for w in ("eff", "param")) + ("rkseq",)
               if _r3("fp_nope_layer_host") else ())                                 # (fr-positions; rkseq: release)
            + (("sin_sig",) if _r3("fp_sinusoid_table") else ())
            + (("max_context", "eff_window", "pos_gate") if USE_CONTEXT else ())    # (round 3: the context gate)
            + (("vocab",) if USE_VOCAB else ()))


# Why each remaining wall has no currency (keyed by slot, then (target, host); "*" = any).
WALL_NOTES = {
    "pos": {("learned_abs", "*"): "absolute table: one RoPE layer provably cannot (Lean rope_cannot_learned_abs: "
                                  "shift-invariant logits); round 2, pass 5: the START-TOKEN rule pays it on a "
                                  "position-free / RoPE host that can zero its frequencies (+2 full layers, +5 / +10 "
                                  "hidden, T + 1 + erasers FFN units; Lean learned_abs_two_block), a wall elsewhere",
            ("alibi", "learned_abs"): "round 2, pass 5: the ramp channel is read through every pre-norm (j / R_j): "
                                      "refuted as exact (proofs/PROOFS.md 'Pre-norm rescue attempts'), not "
                                      "measured approximately -- a wall (verify/REVIEW.md finding 6: its width was "
                                      "never charged)",
            ("rope", "*"): "content-dependent rotation: only a RoPE host (or one that contains it) "
                           "reproduces it (positional_cross: 0.99-1.13 from nope / alibi / learned-abs)",
            ("rope", "partial_rope"): "paid where the host's rotated dims per head hold the target's rotating "
                                      "pairs (rope_fits(rope_sched target, rope_cap host), hf/partial_rope_host.py: exact); "
                                      "below that a wall -- each pair is a distinct position-dependent form, pass "
                                      "dims are position-free and more heads do not add logits",
            ("partial_rope", "rope"): "paid where the host's inv_freq table holds the target's rotating pairs "
                                      "(rope_fits; its pass dims at zero frequency): a one-table host for a "
                                      "target with several RoPE tables or RoPE + NoPE layers needs the room",
            ("partial_rope", "*"): "needs rotating dims: nope / alibi / learned-abs hosts have none",
            ("sinusoidal", "*"): "a FIXED sinusoid table (XGLM, CTRL): a learned-abs host could hold it, but no edge is "
                                 "built yet (round-2 classification pass: a conservative wall)",
            ("*", "sinusoidal"): "a fixed-table host cannot load another table (no edge yet)",
            ("alibi", "nope"): "NoPE has no positional field to carry the ramp",
            ("nope", "alibi"): "alibi-gated: paid only where a config value switches the host's ALiBi off (Falcon's "
                               "alibi flag, the remote MPT's alibi_bias_max); BLOOM / Baichuan-13B / native MPT compute "
                               "their slopes from the head count"},
    "act": {("situ", "*"): "no host within ACT_FLOOR_MAX (best single-op floor: GeLU 0.188)",
            ("relu2", "*"): "only xIELU reaches it (limit); every other host floors at 0.207+",
            ("poly_norm", "*"): "no measured floor and no construction",
            ("*", "*"): "single-op minimax floor above ACT_FLOOR_MAX (act/activation_cross.py): "
                        "not high-agreement generation on real models (act/act_swap_agreement.py); round 2, pass 5: "
                        "paid at FFN width x k where a k-unit fit is within it (graph/act_width.py K tables: SiLU / "
                        "swigluoai <- GeLU x3, GeLU <- SiLU / swigluoai x5 - 6, mish only with a gate bias)"},
    "mixer": {("hybrid", "ssm"): "a pure SSM host has no attention layers",
              ("hybrid", "linear"): "a pure linear host has no attention layers",
              ("ssm", "hybrid"): "SSM families form a one-way lattice (attn/ssm_cross.py); the mixer slot "
                                 "does not carry the family, so a hybrid host's recurrence is not known to "
                                 "carry the target's -- not claimed",
              ("ssm", "linear"): "one-way SSM lattice; family not in the slot -- not claimed",
              ("linear", "ssm"): "one-way SSM lattice; family not in the slot -- not claimed",
              ("linear", "hybrid"): "one-way SSM lattice; family not in the slot -- not claimed",
              ("softmax", "*"): "plain softmax targets: only the host's own-KV attention layers count "
                                "(ZERO_EDGES); a pure SSM / linear host has none",
              ("*", "softmax"): "near-gap not payable (attn/mixer_gap_agreement.py): the softmax "
                                "constructions that keep generation identical need a prefix-sum layer and a "
                                "non-stock pre-norm (data-dependent decay), or 768 - 26,116 heads, or depth ~ "
                                "context (the delta rule); at the edge's price (one layer, fixed decay, few "
                                "heads, a distilled layer) generation diverges"},
    "window": {("full", "local"): "only the host's full-window layers count (ZERO_EDGES); a stack of "
                                  "sliding layers reaches but does not compute a full softmax -- unless the host's "
                                  "window is config its code reads (round 2, pass 5: WINDOW_FREE_TYPES, every sliding "
                                  "layer full)"},
    "crosslayer_kv": {("yes", "no"): "CLA carry not payable (CLA_CARRY_PAYABLE): under a stock pre-norm the "
                                     "shared rms rescales the carried copy (block/prenorm_carry.py, ~6e-2 on "
                                     "stock Llama layers), two orders above ACT_FLOOR_MAX",
                      ("yes", "*"): "a KV-sharing target on a non-plain host: not claimed"},
    "gen": {("diffusion", "ar"): "a causal host cannot see the future in one pass (one-way wall)",
            ("block", "ar"): "a causal host cannot see the rest of a block in one pass (one-way wall)"},
    # round 3 (nonlm_heads): the readout rule (HEAD_ORDER) -- an output read as fixed rows at fixed positions only
    "head": {("lm", "*"): "a non-LM head (score / embedding / value / flow) emits no next-token distribution",
             ("value", "*"): "a ReLU MLP after the final norm: an LM host has no nonlinearity after its last norm",
             ("flow", "*"): "the flow net v(x, t, z) reads an external sample x and time t no LM forward has",
             ("embedding", "lm"): "mean pooling / L2 normalisation are not a fixed read of the host's output (only "
                                  "last- / first-token embeddings without L2 are: _r3_masks_nonlm)",
             ("*", "*"): "a different readout kind: no fixed read of one head gives the other"},
    "input": {("*", "*"): "a float series is not an index into a token table, and a patch MLP cannot do a lookup"},
    "norm": {},
    "qknorm": {("no", "*"): "qk-norm regime not payable (REGIME_PAYABLE): a qk-norm host with a fixed gain normalises "
                            "away the per-token |q|, |k| a plain target uses (attn/regime_agreement.py: 15 - 42% top-1); "
                            "round 2, pass 5: paid by the eps lever (a gained RMS variant, eps > 0) or a config flag "
                            "(QKNORM_SWITCH_TYPES), so a wall only for weight-free / LayerNorm variants",
               ("*", "no"): "qk-norm regime not payable (REGIME_PAYABLE): a fixed-scale fold of the per-token 1/rms(q), "
                            "1/rms(k) diverges on real checkpoints (Qwen3 / OLMo-2 / Gemma-3: top-1 12 - 49%)",
               ("*", "*"): "a different qk-norm variant (QKNORM_VARIANTS): per head vs over the whole projection, before "
                           "vs after RoPE with a gain, RMS vs LayerNorm, shared vs per-head gains compute different "
                           "functions; only a gain / bias setting nests them (QKNORM_NEST)"},
    "softcap": {("attn", "*"): "an attention softcap is not payable without one (REGIME_PAYABLE): Gemma-2-9B's attention "
                               "scores pass the cap (max |z| 66 vs c = 50); without it 75% of continuations stay "
                               "identical"},
    "sink": {("yes", "no"): "a learned sink needs a query-independent logit; dropping it is measured not payable "
                            "(gpt-oss-tinystories: top-1 97.5%, 33% identical); round 2, pass 5: the start-token "
                            "rule pays it on a host with a q bias and a spare zero-frequency slot (+1 hidden)"},
}


def blocked_walls(P):
    """Every (slot, target value, host value) pair present among the points P that NO currency
    crosses under the headline relation: the categorical slots of emul_matrix(mode='payable').
    The ZERO_EDGES pairs are crossed (by depth) for PLAIN softmax targets, so they are listed as
    walls only for the other targets (hybrid / SSM / KV-sharing), and marked so. The ROT-gated pos
    pairs (a RoPE-family target on another RoPE-family host) are crossed only where the host's tables
    hold the target's schedules (rope_fits): listed, and marked so."""
    out = []
    vals = {s: sorted({p["slots"].get(s, SLOT_DEFAULT.get(s)) for p in P}) for s, _ in SLOTS}
    for slot in CAT_SLOTS + ("mixer", "window", "crosslayer_kv"):
        for a in vals[slot]:
            for b in vals[slot]:
                rot_gated = slot == "pos" and a != b and a in ROT_TARGETS and b in ROT_HOSTS
                alibi_gated = slot == "pos" and (a, b) == ("nope", "alibi")      # (emul_matrix: alibi_off)
                if a == b or (payable_reaches(slot, a, b) and not rot_gated and not alibi_gated):
                    continue
                notes = WALL_NOTES.get(slot, {})
                why = notes.get((a, b)) or notes.get((a, "*")) or notes.get(("*", b)) or notes.get(("*", "*")) or ""
                if (slot, a, b) in ZERO_EDGES:
                    why = f"paid in depth for plain softmax targets ({ZERO_EDGES[(slot, a, b)]}); a wall otherwise"
                if rot_gated:
                    why = "rot-gated: " + why
                out.append((slot, a, b, why))
    return out


def _dedup_points(models):
    seen, ms = set(), []
    for m in models:
        if m["name"] in seen or m["dims"]["hidden"] == 0 or m["dims"]["layers"] == 0:
            continue
        seen.add(m["name"])
        ms.append(m)
    # KEY = every axis emulates()/emul_matrix compares, so same-key <=> mutually-emulable and
    # deduping loses nothing. head_dim/qheads/eff_layers/param_layers are pinned by q_cap/kv_cap
    # + the slots in today's zoo (re-deduping with them is a no-op), but list them EXPLICITLY so
    # the dedup stays consistent with the relation if a future model breaks that coupling.
    KD = _point_key_dims()
    pts = {}
    for m in ms:
        key = (tuple(m["slots"][s] for s, _ in SLOTS if s not in _AUX_SLOTS), m["slots"].get("feats", ""),
               tuple(m["dims"][k] for k in KD))
        pts.setdefault(key, {"slots": m["slots"], "dims": m["dims"], "members": []})["members"].append(m["name"])
    P = list(pts.values())
    for p in P:
        p["members"].sort(key=lambda nm: ("tiny" in nm.lower() or "test" in nm.lower(), len(nm), nm))
    return P


# The zoo page (--html-models): graph/templates/zoo.html filled with zoo_data(). Companion to the post.
ZOO_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "zoo.html")
ZOO_SNAPSHOT = "2026-11-10"          # date graph/hub_10k.json was fetched (README)
ZOO_POST_URL = "https://alexlitzenberger.com/blog/the_emulation_frontier"
ZOO_REPO_URL = "https://github.com/alexlitz/the_emulation_frontier"
ZOO_DIMS = ("hidden", "layers", "eff_layers", "param_layers", "kv_cap", "q_cap", "qheads", "head_dim",
            "ffn_total", "ffn_active", "experts", "attn_eff", "full_eff", "attn_param", "full_param")
ZOO_CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hub_10k.json")


def zoo_data(models, snapshot=ZOO_SNAPSHOT, configs=None):
    """Everything the zoo page shows, from the same code paths as FRONTIER.md: one node per distinct
    model shape (_dedup_points), the HEADLINE relation emul_matrix(mode="payable"), the frontier
    representatives (frontier.reps_of) and their bindings (frontier.bindings).

    members = the shape's repos, most downloaded first
    cover   = models the shape emulates, its own repos included (FRONTIER.md's `cover`)
    by      = models in OTHER shapes that emulate it (in-degree), bys = those shapes
    fr      = 1 frontier representative, 2 on the frontier but tied (mutually emulable) with one
    hasse   = the strict relation's covering pairs [target, host] (no strict 2-path between them)
    ties    = mutually emulable pairs [i, j] (distinct shapes that emulate each other)
    emu     = the full relation as a bitset, row j = the shapes host j emulates (np.packbits, base64):
              the relation is not transitive, so the page reads it directly rather than walking edges.
    edges   = the edge view (graph.edge_plan.zoo_edges): the construction of every edge the page can open
              -- the covering (hasse) pairs in order, then every (shape, frontier host) pair not among them,
              shapes ascending, hosts ascending -- with whether elephantine.emulate v1 builds it for the
              shapes' command repos (node "ci": index into members, when not the first).
    configs = repo -> raw config (default: graph/hub_10k.json), for the v1 support check."""
    import base64

    import numpy as np

    from graph.figures import prm, real_prm
    from graph.frontier import bindings, reps_of
    rank = {m["name"]: k for k, m in enumerate(models)}           # snapshot order = downloads
    P = _dedup_points(models)
    n = len(P)
    ok = emul_matrix(P, mode="payable")                         # HEADLINE: capacity + payable type walls
    cap = emul_matrix(P, mode="capacity")
    cnt = np.array([len(p["members"]) for p in P])
    cover = (ok * cnt[:, None]).sum(0)                          # ok[i, j]: j emulates i
    o = ok.copy()
    np.fill_diagonal(o, False)
    by_models, by_shapes = (o * cnt[None, :]).sum(1), o.sum(1)
    strict = o & ~o.T
    reps = reps_of(ok, P)
    rep_set = set(int(i) for i in reps)
    front = np.where(~strict.any(1))[0]
    why = bindings(P, ok, cap, reps)
    s2 = (strict.astype(np.int32) @ strict.astype(np.int32)) > 0
    hasse = strict & ~s2
    nodes = []
    for i, p in enumerate(P):
        fr, bind = 0, ""
        if i in rep_set:
            fr, bind = 1, why[i]
        elif i in front:
            r = next(int(j) for j in reps if ok[i, j] and ok[j, i])
            fr, bind = 2, f"TIE with {short(P[r]['members'][0])} (mutually emulable; counted once)"
        pop = sorted(p["members"], key=lambda m: rank.get(m, len(rank)))
        size = real_prm(p)                    # placement only; a published count far below the config's
        if size < prm(p["dims"]) / 20:        # own estimate is a bad safetensors total: use the estimate
            size = prm(p["dims"])
        # frontier shapes keep FRONTIER.md's name; any other shape is named after its most-downloaded repo
        nodes.append({"name": short(p["members"][0] if fr == 1 else pop[0]), "members": pop,
                      "slots": [p["slots"][k] for k, _ in SLOTS],
                      "dims": [int(p["dims"].get(k, 0) or 0) for k in ZOO_DIMS],
                      "params": float(size), "cover": int(cover[i]), "by": int(by_models[i]),
                      "bys": int(by_shapes[i]), "fr": fr, "bind": bind})
    nd = sum(1 for i in reps if P[i]["slots"]["gen"] != "ar")      # diffusion + block-causal
    # the edge view: covering pairs, then (shape, frontier host) pairs
    from graph.edge_plan import cmd_repo, zoo_edges
    if configs is None:
        configs = dict(iter_json_configs(ZOO_CONFIGS))
    pairs = [(int(a), int(b)) for a, b in zip(*np.where(hasse))]
    in_hasse = set(pairs)
    rep_sorted = sorted(rep_set)
    pairs += [(i, j) for i in range(n) for j in rep_sorted if o[i, j] and (i, j) not in in_hasse]
    repos = []
    for k, nd_ in enumerate(nodes):
        r = cmd_repo(nd_["members"], configs)
        repos.append(r)
        if r != nd_["members"][0]:
            nd_["ci"] = nd_["members"].index(r)
    exact = emul_matrix(P, mode="exact")
    edges = zoo_edges(P, exact, cap, pairs, repos, configs)
    bits = np.packbits(o.T, axis=1)                             # row j = column j of ok
    return {"snapshot": snapshot, "post": ZOO_POST_URL, "repo": ZOO_REPO_URL,
            "n_models": int(cnt.sum()), "n_shapes": n, "n_frontier": len(reps), "n_frontier_diff": nd,
            "n_frontier_capacity": len(reps_of(cap, P)),
            "slot_keys": [k for k, _ in SLOTS], "dim_keys": list(ZOO_DIMS), "nodes": nodes,
            "hasse": [[int(a), int(b)] for a, b in zip(*np.where(hasse))],
            "ties": [[int(a), int(b)] for a, b in zip(*np.where(np.triu(o & o.T)))],
            "emu": base64.b64encode(bits.tobytes()).decode(), "emu_row": int(bits.shape[1]),
            "edges": edges}


def export_html_points(models, path, configs=None):
    """The zoo page: a self-contained HTML file, one node per distinct model shape."""
    import json
    data = zoo_data(models, configs=configs)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(ZOO_TEMPLATE) as f:
        html = f.read().replace("__ZOO_DATA__", blob)
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path}: {data['n_shapes']} model shapes ({data['n_models']} models), "
          f"{len(data['hasse'])} covering edges, {data['n_frontier']} on the frontier "
          f"({len(html) / 1e3:.0f} kB)")


def analyze(models):
    """Capacity-aware, vectorized: class collapse + minimal REAL emulator frontier."""
    import numpy as np
    seen, ms = set(), []
    for m in models:
        if m["name"] in seen or m["dims"]["hidden"] == 0 or m["dims"]["layers"] == 0:
            continue
        seen.add(m["name"])
        ms.append(m)
    n = len(ms)

    # class histogram (the collapse: N models -> few architectures)
    cls = {}
    for m in ms:
        cls.setdefault(tuple(m["slots"][s] for s, _ in SLOTS), []).append(m["name"])
    moe = sum(1 for m in ms if m["slots"]["ffn"] in ("moe", "moe_shared"))
    print(f"\n=== {n} unique models -> {len(cls)} architecture classes "
          f"({moe} MoE / {n - moe} dense) ===")
    slotnames = [s for s, _ in SLOTS]
    for k, mem in sorted(cls.items(), key=lambda kv: -len(kv[1]))[:14]:
        print(f"  {len(mem):5}x  {slot_str(dict(zip(slotnames, k)))}   e.g. {short(sorted(mem, key=len)[0])}")

    # dedup to (slots, dims) POINTS, then Pareto frontier over points (vectorized)
    # KEY = every axis emulates()/emul_matrix compares, so same-key <=> mutually-emulable and
    # deduping loses nothing. head_dim/qheads/eff_layers/param_layers are pinned by q_cap/kv_cap
    # + the slots in today's zoo (re-deduping with them is a no-op), but list them EXPLICITLY so
    # the dedup stays consistent with the relation if a future model breaks that coupling.
    KD = _point_key_dims()
    pts = {}
    for m in ms:
        key = (tuple(m["slots"][s] for s, _ in SLOTS if s not in _AUX_SLOTS), m["slots"].get("feats", ""),
               tuple(m["dims"][k] for k in KD))
        pts.setdefault(key, {"slots": m["slots"], "dims": m["dims"], "members": []})["members"].append(m["name"])
    P = list(pts.values())
    for p in P:
        p["members"].sort(key=lambda nm: ("tiny" in nm.lower() or "test" in nm.lower(), len(nm), nm))
    ok = emul_matrix(P, mode="payable")                         # HEADLINE: capacity + payable type walls
    counts = np.array([len(p["members"]) for p in P])
    strict = ok & ~ok.T
    np.fill_diagonal(strict, False)
    front_idx = np.where(~strict.any(1))[0]

    # collapse mutually-emulating frontier points to one representative
    reps, used = [], set()
    for i in front_idx:
        if i in used:
            continue
        eq = [j for j in front_idx if ok[i, j] and ok[j, i]]
        used.update(eq)
        reps.append(min(eq, key=lambda j: ("tiny" in P[j]["members"][0].lower(),
                                            len(P[j]["members"][0]))))
    cover = {i: int(counts[ok[:, i]].sum()) for i in reps}
    reps.sort(key=lambda i: -cover[i])
    print(f"\n=== capacity-aware emulator frontier: {len(reps)} REAL models emulate all {n} ===")
    for i in reps[:22]:
        s, d = P[i]["slots"], P[i]["dims"]
        print(f"  {short(P[i]['members'][0]):30} [{s['norm'][:3]} {s['pos'][:4]} {s['kv']:3} "
              f"{s['ffn']:10} {s['act'][:4]}] h{d['hidden']} L{d['layers']} "
              f"kv{d['kv_cap']} ffnT{d['ffn_total']} -> emulates {cover[i]}")
    if len(reps) > 22:
        print(f"  ... (+{len(reps) - 22} more frontier models)")

    # non-trivial mutual subsets among POINTS (distinct arch, mutual emulation)
    mut = ok & ok.T
    np.fill_diagonal(mut, False)
    kinds = {}
    for i, j in zip(*np.where(np.triu(mut))):
        if P[i]["slots"] != P[j]["slots"]:
            diff = tuple(sorted(s for s, _ in SLOTS if P[i]["slots"][s] != P[j]["slots"][s]))
            kinds.setdefault(diff, (short(P[i]["members"][0]), short(P[j]["members"][0]),
                                    P[i]["slots"], P[j]["slots"]))
    total_mut = int(np.triu(mut).sum())
    print(f"\n=== non-trivial mutual subsets: {total_mut} point-pairs, {len(kinds)} kinds ===")
    for diff, (na, nb, s1, s2) in kinds.items():
        print(f"  differ in [{', '.join(f'{s}:{s1[s]}~{s2[s]}' for s in diff)}]  e.g. {na} <-> {nb}")
    if not kinds:
        print("  none among distinct architectures (real sizes never match exactly).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="graph/model_subset_graph")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/huggingface/hub"))
    ap.add_argument("--from-json", default=None, help="hub_configs.json from fetch_configs.py")
    ap.add_argument("--no-local", action="store_true", help="skip the local HF cache")
    ap.add_argument("--min-members", type=int, default=1, help="drop classes with fewer members")
    ap.add_argument("--abstract", action="store_true",
                    help="add the abstract universal-parent (supremum) node")
    ap.add_argument("--analyze", action="store_true",
                    help="print MoE breakdown + minimal emulator set + mutual subsets")
    ap.add_argument("--dump", default=None, help="write full checkpoint lists (per class) to this path")
    ap.add_argument("--html", default=None, help="interactive class-level HTML graph to this path")
    ap.add_argument("--html-models", default=None,
                    help="interactive INSTANCE-level HTML (one node per distinct model spec)")
    ap.add_argument("--moe-depth-factor", type=float, default=2.0,
                    help="depth multiplier a dense FFN needs to emulate a MoE (default 2)")
    ap.add_argument("--with-vocab", action="store_true",
                    help="count vocab_size as a capacity axis (B needs vocab_B >= vocab_A); off by default")
    ap.add_argument("--with-context", action="store_true",
                    help="price max-context as a DEMAND currency + the tight window<->depth<->context "
                         "coupling (a local host reaches a wider target only by paying depth); off by default")
    ap.add_argument("--no-norm-fold", action="store_true",
                    help="treat LayerNorm/RMSNorm as incomparable (the fold needs a global "
                         "cross-layer reparam, not a local swap)")
    ap.add_argument("--no-approx-act", action="store_true",
                    help="drop every activation FLOOR edge, keeping only the exact folds and the "
                         "beta -> inf limits (the same relation as the calibrated ACT_FLOOR_MAX = 0.0005: "
                         "no measured floor lies in (0, 0.009))")
    args = ap.parse_args()
    global MOE_DEPTH_FACTOR, USE_VOCAB, USE_CONTEXT
    MOE_DEPTH_FACTOR = args.moe_depth_factor
    USE_VOCAB = args.with_vocab
    USE_CONTEXT = args.with_context
    if args.no_norm_fold:
        NORM_ORDER.clear()
        NORM_ORDER.update({"LayerNorm": [("AffineNorm", "fold", True)],
                           "RMSNorm": [("AffineNorm", "free", True)]})
        REACH["norm"] = reach(NORM_ORDER)
    if args.no_approx_act:                       # only the exact folds and the beta -> inf limits
        ACT_ORDER.clear()
        ACT_ORDER.update(_act_order(floor_max=0.0))
        REACH["act"] = one_hop(ACT_ORDER)

    items = []
    if not args.no_local:
        items += list(iter_local_configs(args.cache))
    if args.from_json:
        items += list(iter_json_configs(args.from_json))
    classes, n_seen = build_classes(items)
    if args.min_members > 1:
        classes = {k: v for k, v in classes.items() if len(v["members"]) >= args.min_members}
    n_models = sum(len(v["members"]) for v in classes.values())
    print(f"sources: {n_seen} configs seen -> {n_models} unique attention-transformer "
          f"models in {len(classes)} architecture classes\n")
    if args.abstract:
        classes[tuple(TOP[s] for s, _ in SLOTS)] = {
            "slots": {s: TOP[s] for s, _ in SLOTS}, "members": ["elephantine (universal parent)"]}
    for k in sorted(classes, key=lambda k: (classes[k]["slots"]["ffn"], classes[k]["slots"]["norm"])):
        s = classes[k]["slots"]
        n = len(classes[k]["members"])
        mem = ", ".join(short(m) for m in classes[k]["members"][:3])
        extra = f" (+{n-3})" if n > 3 else ""
        print(f"  {n:>4}x [{s['norm']:9} {s['pos']:11} {s['kv']:4} {s['ffn']:10} {s['act']:6}]  {mem}{extra}")

    dot, edges, cover = build_dot(classes)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dot_path = args.out + ".dot"
    with open(dot_path, "w") as f:
        f.write(dot)
    print(f"\n{len(edges)} subset relations, {len(cover)} covering edges. wrote {dot_path}")
    try:
        subprocess.run(["dot", "-Tsvg", dot_path, "-o", f"{args.out}.svg"],
                       check=True, capture_output=True)
        print(f"  rendered {args.out}.svg")
    except Exception as e:
        print(f"  (svg render skipped: {e})")

    print("\ncovering edges (A is a special case of B):")
    for (a, b), diffs in sorted(cover.items(), key=lambda kv: str(kv[0])):
        A, B = classes[a]["slots"], classes[b]["slots"]
        cur = ", ".join(f"{s}:{'/'.join(cost[0]) if cost else '='}{'' if not cost or cost[1] else '~'}"
                        for (s, cost) in diffs)
        print(f"  {short(classes[a]['members'][0]):22} -> {short(classes[b]['members'][0]):22}  [{cur}]")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write(f"# {n_models} models in {len(classes)} architecture classes "
                    "(norm | pos | kv | ffn | act)\n\n")
            for k in sorted(classes, key=lambda k: -len(classes[k]["members"])):
                f.write(f"## {len(classes[k]['members'])}x  {slot_str(classes[k]['slots'])}\n")
                for mname in sorted(classes[k]["members"]):
                    f.write(f"  {mname}\n")
                f.write("\n")
        print(f"wrote {args.dump}: {n_models} checkpoints in {len(classes)} classes")

    if args.html:
        export_html(classes, args.html)

    if args.html_models or args.analyze:
        models = []
        for name, d in items:
            if _junky(name):
                continue
            c = classify(d)
            if c is not None:
                models.append({"name": name, "slots": c, "dims": dims_of(d)})
        if args.html_models:
            export_html_points(models, args.html_models, configs=dict(items))
        if args.analyze:
            analyze(models)


if __name__ == "__main__":
    main()
