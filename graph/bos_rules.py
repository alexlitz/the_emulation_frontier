"""Graph rules for the start-marker (BOS) assumption: pure functions and constants.

PORTED in round 2, pass 5: graph/model_graph.emul_matrix applies both rules (BOS_* / SINK_* there, dims q_bias /
pos_table), with the marker seen by the EMULATOR only (T + 1 = the target's context + the marker) and the shared
unit count; apply_rules / estimate below re-apply them on top of the relation (now a no-op check).

The assumption: every input begins with ONE fixed special token (a BOS-like marker, the same token for
every input). Two walls depend on it. Both constructions are exact (Lean: proofs/lean,
`learned_abs_two_block` / `learned_abs_model` and `sink_sub_plain`) and are built on stock modules in
hf/bos_constructions.py.

1. learned_abs target <- position-free or RoPE-family host (today a wall: POS_ORDER["learned_abs"]
   reaches only pos_top). The host runs with its positional field at the identity: RoPE / partial
   RoPE through inv_freq = 0 (hf/nope_rope.py), or no positional field at all. ALiBi hosts count only
   under the graph's nope <= alibi edge (BOS_LEARNED_ABS_HOSTS_ALIBI): stock BLOOM / MPT / Falcon have
   no slope buffer to zero. Charges, with T + 1 = the target's max context (the marker at position 0
   included):
     depth   +2 layers (BOS_EXTRA_LAYERS), both with a FULL attention window (block 1 must see the
             marker from every position);
     width   +5 residual channels on an RMSNorm host (s, b, kappa, u, v), and +10 on a LayerNorm host:
             the five are mirrored [c; -c] so the residual stays mean-zero and the host's centring is
             a no-op;
     FFN     per layer, the units of the two extra MLPs:
               LEAN (the proof as written)   block 1 T + 3, block 2 5 + d (T + 3)
               SHARED (hf/bos_constructions)  block 1 T + 1, block 2 T + 1 + erasers.
                 The erasers are 5 on a gated host. On a vanilla host they are 10, two units per channel
                 (act(z) - act(-z) = z), and only for x F(x) activations.
             The shared count is sound in the proof's own terms: the Lean's kinks relu(u - t_i) are the
             same functions for every content coordinate, and only their output columns differ. Any
             activation whose large-scale limit is a ramp or step works at a FINITE scale. The knot
             system H c = Y is invertible because its ReLU limit is triangular, so this is exact and
             not a limit. A vanilla host needs ReLU or an x F(x) activation (SiLU, GELU, swigluoai)
             for the exact linear erasers.
   Exact in real arithmetic, and measured at fp64 rounding of the construction's sums. fp32 is limited
   by the interpolation's conditioning (terms ~ T^2 |dP|): see hf/bos_constructions.py and
   tests/test_bos_constructions.py.

2. sink target (gpt-oss: learned per-head softmax sinks) <- sink-free host (today free both ways:
   SINK_ORDER, and `sink` is an _AUX_SLOT left out of the dedup key). The marker is a host-only token
   at position 0. It supplies, in every head, a key with logit s_h for every query and a zero value.
   Requirements:
     a q bias       a constant query component. A k bias alone puts the constant on every key, and
                    softmax cancels it. A reserved channel is not constant after RMSNorm
                    (`prenorm_readout_obstruction`);
     a spare pair   one more head dim at frequency 0 for the constant (`sink_needs_zero_freq`). On a
                    rotating host that is a whole pair, so +2 head dims (+1 on a position-free host),
                    and q_cap / kv_cap grow with the head dim;
     +1 hidden      a reserved channel sigma, nonzero only at the marker, so the marker's key and value
                    can be set per layer (its residual is fixed: it sees only itself).
   Exact for full-attention layers. In a sliding-window layer the marker leaves the window after
   `window` tokens, so a windowed sink target is exact only up to its window. Measured on
   gpt-oss-tinystories: fp64 max |dlogit| ~ 1e-14.

WIRING (after the merge). In emul_matrix, the learned-abs rows whose host passes bos_learned_abs_ok
are compared against bos_learned_abs_target_dims, with the target's pos read as "nope". The sink rule
needs `sink` taken out of _AUX_SLOTS (so it enters the dedup key) and a q-bias flag in dims_of (has_q_bias)
and in _point_key_dims. A sink target on a sink-free host is then ok only if bos_sink_ok holds, and it
is compared against sink_target_dims. apply_rules below does exactly this on top of the current matrix.

ESTIMATE (`python -m graph.bos_rules`, snapshot 2026-11-10, headline 104):
  learned_abs, Lean or shared charges, RoPE / NoPE hosts   104, no change. Three of the four learned-abs
      frontier models are GELU (codeparrot, granite-20b-code-8k, ruGPT-3.5-13B), and the GELU <- SiLU
      floor still blocks every RoPE host big enough. opt-66b is a capacity record (MHA kv_cap 9,216).
  + ALiBi hosts (the graph's nope <= alibi)                100, all four leave (to BLOOM / Poro-34B).
  sink rule                                                105: openai/gpt-oss-120b enters. Its only
      dominating host, DeepSeek-V4-Flash, has no q bias.
"""
from __future__ import annotations

# --- constants ---------------------------------------------------------------------------------------

BOS_EXTRA_LAYERS = 2
BOS_EXTRA_HIDDEN = {"RMSNorm": 5, "LayerNorm": 10}      # LayerNorm host: the 5 channels mirrored
# hosts whose positional field is switched off by a buffer: RoPE / partial RoPE with inv_freq = 0
# (hf/nope_rope.py), or none at all. ALiBi is NOT in the default set: stock BLOOM / MPT / Falcon build
# the slopes from num_heads inside forward (no buffer to zero), although the graph's POS_ORDER treats
# nope <= alibi as free. BOS_LEARNED_ABS_HOSTS_ALIBI is the set under that graph edge.
BOS_LEARNED_ABS_HOSTS = frozenset({"nope", "rope", "partial_rope"})
BOS_LEARNED_ABS_HOSTS_ALIBI = BOS_LEARNED_ABS_HOSTS | {"alibi"}
# activations whose large-scale limit is a ramp or step (ReLU, the swish family, xIELU / relu2 ramps)
BOS_KNOT_ACTS = frozenset({"ReLU", "SiLU", "GeLU", "mish", "swigluoai", "relu2", "xielu", "situ"})
# vanilla (ungated) hosts: an exact linear unit needs act(z) - act(-z) = z
BOS_ODD_LINEAR_ACTS = frozenset({"ReLU", "SiLU", "GeLU", "swigluoai"})
GATED_FFN = frozenset({"glu", "moe", "moe_shared"})

SINK_EXTRA_HIDDEN = 1
SINK_SPARE_HEAD_DIMS = {"rope": 2, "partial_rope": 2, "nope": 1, "alibi": 1, "learned_abs": 1}

# configs: explicit q-bias keys, then the model types whose q projection always has a bias
_QBIAS_KEYS = ("attention_bias", "qkv_bias", "use_qkv_bias", "add_qkv_bias", "enable_bias", "attn_bias",
               "use_bias", "bias")
QBIAS_TYPES = frozenset({"qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl", "qwen", "gpt2", "gpt_bigcode",
                         "opt", "bloom", "gpt_neox", "phi", "starcoder2", "xglm", "gpt_oss",
                         "gpt_oss_puzzle", "glm", "chatglm"})


# --- learned_abs <- position-free / RoPE host -------------------------------------------------------

def bos_learned_abs_units(d, T, shared=False, gated=True):
    """(block-1 units, block-2 units) of the two extra MLPs: d = target hidden, T + 1 = its context."""
    if shared:
        return T + 1, T + 1 + (5 if gated else 10)
    return T + 3, 5 + d * (T + 3)


def bos_learned_abs_charges(target_dims, host_slots, shared=False):
    """The capacity a host pays for a learned-abs target, beyond what the target itself needs. Round 3
    (max_context_unknown): an unknown context (None) has no charge to compute -- callers skip it (the rule is not
    credited); asking for one is an error, never a default."""
    if target_dims.get("max_context", 0) is None:
        raise ValueError("learned-abs target with an unknown context: no table length to charge")
    T = max(int(target_dims.get("max_context") or 0) - 1, 1)
    gated = host_slots.get("ffn") in GATED_FFN
    u1, u2 = bos_learned_abs_units(int(target_dims["hidden"]), T, shared, gated)
    return dict(layers=BOS_EXTRA_LAYERS, hidden=BOS_EXTRA_HIDDEN.get(host_slots.get("norm"), 5),
                ffn_units=max(u1, u2), T=T)


def bos_learned_abs_host_ok(host_slots, host_dims=None, hosts=BOS_LEARNED_ABS_HOSTS):
    """The non-capacity conditions on the HOST: a positional field it can switch off, a pre-norm it can
    read at a constant rms, an activation the knot system works with, softmax attention, and at least
    one full-window attention layer for block 1."""
    if host_slots.get("pos") not in hosts:
        return False
    if host_slots.get("norm") not in BOS_EXTRA_HIDDEN:
        return False
    act, ffn = host_slots.get("act"), host_slots.get("ffn")
    if act not in BOS_KNOT_ACTS:
        return False
    if ffn not in GATED_FFN and act not in BOS_ODD_LINEAR_ACTS:
        return False
    if host_slots.get("mixer", "softmax") not in ("softmax", "hybrid"):
        return False
    if host_slots.get("window") == "local" and host_dims is not None and not host_dims.get("full_eff", 0):
        return False
    return True


def bos_learned_abs_ok(target_dims, host_dims, host_slots, shared=False, hosts=BOS_LEARNED_ABS_HOSTS):
    """A learned-abs target on a host whose positional field is not learned-abs: the host conditions and
    the charges against the host's capacity (the rest of emul_matrix's capacity comparison still
    applies). Uses the host's full-window depth when it has sliding layers."""
    if not bos_learned_abs_host_ok(host_slots, host_dims, hosts):
        return False
    if target_dims.get("max_context", 0) is None:      # (round 3: an unknown table length is not charged, not credited)
        return False
    c = bos_learned_abs_charges(target_dims, host_slots, shared)
    depth = host_dims.get("full_eff", host_dims["eff_layers"]) if host_slots.get("window") == "local" \
        else host_dims["eff_layers"]
    return (host_dims["hidden"] >= target_dims["hidden"] + c["hidden"]
            and depth >= target_dims["eff_layers"] + c["layers"]
            and host_dims["param_layers"] >= target_dims["param_layers"] + c["layers"]
            and host_dims["ffn_active"] >= max(target_dims["ffn_active"], c["ffn_units"]))


def bos_learned_abs_target_dims(target_dims, host_slots, shared=False):
    """The target's dims with the charges added: what emul_matrix's capacity comparison should see
    when the host crosses the learned-abs wall with the marker."""
    if target_dims.get("max_context", 0) is None:      # (round 3: unknown length; bos_learned_abs_ok refuses it)
        return dict(target_dims)
    c = bos_learned_abs_charges(target_dims, host_slots, shared)
    out = dict(target_dims)
    for k in ("eff_layers", "param_layers", "layers", "attn_eff", "attn_param", "full_eff", "full_param"):
        if k in out:
            out[k] = out[k] + c["layers"]
    out["hidden"] = out["hidden"] + c["hidden"]
    out["ffn_active"] = max(out["ffn_active"], c["ffn_units"])
    out["ffn_total"] = max(out["ffn_total"], c["ffn_units"])
    return out


# --- sink <- sink-free host --------------------------------------------------------------------------

def has_q_bias(config):
    """Does this config's attention have a query bias? Explicit keys first, then the model type."""
    try:
        from graph.model_graph import _lm_config
        d = _lm_config(config)
    except Exception:                                       # noqa: BLE001 -- a bare dict is fine too
        d = config
    for k in _QBIAS_KEYS:
        if k in d and isinstance(d[k], bool):
            return d[k]
    if isinstance(d.get("no_bias"), bool) and d.get("model_type") not in ("qwen",):
        return not d["no_bias"]
    return d.get("model_type") in QBIAS_TYPES


def bos_sink_ok(target_dims, host_dims, host_slots, host_q_bias):
    """A sink target on a host. A sink host is free, as SINK_ORDER has it. A sink-free host needs a
    q bias, a spare zero-frequency head dim pair (+2 head dims on a rotating host, +1 on a
    position-free one) and one reserved residual channel. The rest is emul_matrix's capacity
    comparison against sink_target_dims."""
    if host_slots.get("sink") == "yes":
        return True
    if not host_q_bias:
        return False
    spare = SINK_SPARE_HEAD_DIMS.get(host_slots.get("pos"))
    if spare is None:
        return False
    return (host_dims["head_dim"] >= target_dims["head_dim"] + spare
            and host_dims["hidden"] >= target_dims["hidden"] + SINK_EXTRA_HIDDEN)


def sink_target_dims(target_dims, host_slots):
    """The sink target's dims with the charges added (head dims, hence q_cap / kv_cap, and hidden)."""
    spare = SINK_SPARE_HEAD_DIMS.get(host_slots.get("pos"), 2)
    out = dict(target_dims)
    hd = out["head_dim"]
    if hd:
        out["q_cap"] = out["q_cap"] * (hd + spare) // hd
        out["kv_cap"] = out["kv_cap"] * (hd + spare) // hd
    out["head_dim"] = hd + spare
    out["hidden"] = out["hidden"] + SINK_EXTRA_HIDDEN
    return out


# --- the headline estimate ---------------------------------------------------------------------------

def _points(json_path):
    """Dedup the snapshot like _dedup_points, but with the sink slot and the q-bias flag in the key."""
    from graph.model_graph import (SLOTS, _junky, _point_key_dims, classify, dims_of,
                                   iter_json_configs)
    KD = _point_key_dims()
    pts = {}
    for n, c in iter_json_configs(json_path):
        if _junky(n):
            continue
        cl = classify(c)
        if not cl:
            continue
        dm = dims_of(c)
        if not (dm["hidden"] and dm["layers"]):
            continue
        qb = has_q_bias(c)
        key = (tuple(cl[s] for s, _ in SLOTS), tuple(dm[k] for k in KD), qb,
               dm["max_context"] if cl["pos"] == "learned_abs" else 0)
        p = pts.setdefault(key, {"slots": cl, "dims": dm, "members": [], "qbias": qb})
        if n not in p["members"]:
            p["members"].append(n)
    P = list(pts.values())
    for p in P:
        p["members"].sort(key=lambda nm: ("tiny" in nm.lower() or "test" in nm.lower(), len(nm), nm))
    return P


def _with_target_rows(P, rows, mod_slots, mod_dims):
    """ok[i, j] for the target rows `rows` re-described as (mod_slots(i), mod_dims(i, j-independent)),
    against every original point as host: emul_matrix on P + the modified copies."""
    import numpy as np

    from graph.model_graph import emul_matrix
    extra = [{"slots": mod_slots(P[i]), "dims": mod_dims(P[i]), "members": P[i]["members"]} for i in rows]
    ok = emul_matrix(P + extra, mode="payable")
    n = len(P)
    return np.asarray(ok[n:, :n])


def apply_rules(P, shared=False, sink_rule=True, learned_abs_rule=True, hosts=BOS_LEARNED_ABS_HOSTS):
    """The headline matrix with the two marker rules applied (emul_matrix(mode='payable') otherwise)."""
    import numpy as np

    from graph.model_graph import emul_matrix
    ok = emul_matrix(P, mode="payable").copy()
    if learned_abs_rule:
        rows = [i for i, p in enumerate(P) if p["slots"]["pos"] == "learned_abs"]
        if rows:
            for norm in ("RMSNorm", "LayerNorm"):
                hs = {"norm": norm, "ffn": "glu"}
                sub = _with_target_rows(P, rows, lambda p: {**p["slots"], "pos": "nope"},
                                        lambda p: bos_learned_abs_target_dims(p["dims"], hs, shared))
                for r, i in enumerate(rows):
                    for j, h in enumerate(P):
                        if h["slots"]["pos"] == "learned_abs" or h["slots"]["norm"] != norm:
                            continue
                        if sub[r, j] and bos_learned_abs_ok(P[i]["dims"], h["dims"], h["slots"], shared, hosts):
                            ok[i, j] = True
    if sink_rule:
        rows = [i for i, p in enumerate(P) if p["slots"]["sink"] == "yes"]
        if rows:
            for pos in ("rope", "nope"):
                sub = _with_target_rows(P, rows, lambda p: {**p["slots"], "sink": "no"},
                                        lambda p: sink_target_dims(p["dims"], {"pos": pos}))
                for r, i in enumerate(rows):
                    for j, h in enumerate(P):
                        if h["slots"]["sink"] == "yes":
                            continue
                        rot = h["slots"]["pos"] in ("rope", "partial_rope")
                        if rot != (pos == "rope"):
                            continue
                        ok[i, j] = bool(sub[r, j]) and bos_sink_ok(P[i]["dims"], h["dims"], h["slots"],
                                                                   h["qbias"])
    np.fill_diagonal(ok, True)
    return ok


def frontier_names(P, ok):
    from graph.frontier import reps_of
    return {P[i]["members"][0] for i in reps_of(ok, P)}


def estimate(json_path="graph/hub_10k.json"):
    from graph.model_graph import emul_matrix
    P = _points(json_path)
    base = emul_matrix(P, mode="payable")
    out = {"points": len(P), "models": sum(len(p["members"]) for p in P)}
    f0 = frontier_names(P, base)
    out["baseline"] = len(f0)
    for label, kw in (("learned_abs (Lean charges)", dict(shared=False, sink_rule=False)),
                      ("learned_abs (shared units)", dict(shared=True, sink_rule=False)),
                      ("learned_abs (shared units, + ALiBi hosts)",
                       dict(shared=True, sink_rule=False, hosts=BOS_LEARNED_ABS_HOSTS_ALIBI)),
                      ("sink rule", dict(learned_abs_rule=False)),
                      ("both (Lean charges)", dict(shared=False)),
                      ("both (shared units)", dict(shared=True))):
        ok = apply_rules(P, **kw)
        f = frontier_names(P, ok)
        idx = {p["members"][0]: i for i, p in enumerate(P)}
        size = lambda j: P[j]["dims"]["hidden"] * P[j]["dims"]["eff_layers"]

        def smallest(i, m):            # the smallest other point that emulates point i under m
            js = [j for j in range(len(P)) if m[i, j] and j != i and not m[j, i]]
            return P[min(js, key=size)]["members"][0] if js else None
        out[label] = dict(frontier=len(f),
                          leave={n: smallest(idx[n], ok) for n in sorted(f0 - f)},
                          enter={n: f"was emulated by {smallest(idx[n], base)}" for n in sorted(f - f0)})
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(estimate(), indent=1))
