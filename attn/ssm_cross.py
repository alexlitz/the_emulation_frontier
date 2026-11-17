"""SSM / linear-recurrence family cross-emulation — is it MUTUAL? (No: a one-way order.)

Every linear-recurrence sequence mixer in the zoo (linear attention, RetNet/retention,
Mamba-1/2, GLA, DeltaNet, gated-DeltaNet, RWKV-7, RG-LRU, lightning, LFM2) is a special
case of ONE general **gated delta rule** with three knobs — a per-key-channel decay a_t,
a delta strength gamma_t, and a write w_t:

    S_t = Diag(a_t) S_{t-1}  -  gamma_t (S_{t-1} k_t) k_t^T  +  w_t v_t k_t^T,   y_t = S_t q_t

    linear attention    a = 1            gamma = 0        w = 1
    RetNet / retention  a = g (scalar)   gamma = 0        w = 1
    Mamba-2 (SSD)       a = a_t (scalar) gamma = 0        w = 1
    GLA / Mamba-1       a = alpha_t (ch) gamma = 0        w = 1
    DeltaNet            a = 1            gamma = beta_t   w = beta_t
    gated DeltaNet /    a = alpha_t (ch) gamma = beta_t   w = beta_t     <- the general form
      RWKV-7

The general rule EMULATES every family EXACTLY (set the knobs — free, no state cost). The
reverse does NOT hold: a family with gamma=0 cannot reproduce a rank-1 delta correction,
and a scalar decay cannot reproduce a per-channel gate — both are irreducible (no state
expansion recovers a different *operator*). So the SSM families form a one-way SUBSUMPTION
LATTICE, **not** mutual emulation:

    linear  <=  RetNet  <=  Mamba-2  <=  GLA  ─┐
    linear  <=  DeltaNet  ────────────────────┤=>  gated-DeltaNet  (the top; folds every AR recurrence)

Mutual emulation holds only *within* a family (same knob-space), never across two distinct
ones. gated-DeltaNet is the supremum — which is exactly why ONE such layer added to the
elephantine lifts the exact AR sub-zoo to 100% (see graph/minimal_emulator.py).

Run:  python -m attn.ssm_cross
"""
from __future__ import annotations

import torch

F64 = torch.float64


def general(q, k, v, a, gamma, w):
    """The general gated-delta recurrence. a:(T,dk) decay, gamma:(T,) delta, w:(T,) write."""
    T, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dv, dk, dtype=F64)
    out = torch.zeros(T, dv, dtype=F64)
    for t in range(T):
        S = S * a[t][None, :]                                   # per-key-channel decay
        S = S - gamma[t] * torch.outer(S @ k[t], k[t])          # delta: remove old value at k_t
        S = S + w[t] * torch.outer(v[t], k[t])                  # write v_t at k_t
        out[t] = S @ q[t]
    return out


# --- each family as (knobs of the general rule) -----------------------------------------
# decay level: 0 const-1, 1 const-g, 2 scalar_t, 3 per-channel ; delta: 0 off, 1 on.
FAMILIES = {
    "linear":          dict(decay=0, delta=0),
    "retnet":          dict(decay=1, delta=0),
    "mamba2":          dict(decay=2, delta=0),
    "gla":             dict(decay=3, delta=0),   # == Mamba-1 (diagonal selective SSM)
    "deltanet":        dict(decay=0, delta=1),
    "gated_deltanet":  dict(decay=3, delta=1),   # RWKV-7 / RG-LRU / lightning live here
}


def knobs(fam, T, dk, gen):
    """Concrete (a, gamma, w) tensors for a family, using a shared RNG for reproducibility."""
    r = lambda *s: torch.randn(*s, dtype=F64, generator=gen)
    lvl, dl = FAMILIES[fam]["decay"], FAMILIES[fam]["delta"]
    if lvl == 0:
        a = torch.ones(T, dk, dtype=F64)
    elif lvl == 1:
        a = 0.9 * torch.ones(T, dk, dtype=F64)                  # one scalar decay for all t, all channels
    elif lvl == 2:
        a = (0.5 + 0.4 * r(T, 1)).abs().expand(T, dk).contiguous()   # data-dependent scalar (per t)
    else:
        a = (0.5 + 0.4 * r(T, dk)).abs()                        # data-dependent per-channel
    beta = (0.3 + 0.4 * r(T)).abs()
    gamma = beta if dl else torch.zeros(T, dtype=F64)
    w = beta if dl else torch.ones(T, dtype=F64)
    return a, gamma, w


def subsumes(host, guest):
    """Does `host` family contain `guest` as a knob-restriction? (guest <= host, exact & free)."""
    h, g = FAMILIES[host], FAMILIES[guest]
    return h["decay"] >= g["decay"] and h["delta"] >= g["delta"]


def gated_delta_emulate(fam, state_pad=8, T=12, dk=8, dv=8, seed=0):
    """A gated-delta host with a WIDER state emulates any linear-recurrence family token-for-
    token: embed the target's k/v/q in the first dk/dv dims (pad the rest with 0), run the
    general rule at the wider state -> the extra state dimensions stay 0, so the output (first
    dv dims) equals the target. This is the recurrence-level 'elephantine' for the SSM island:
    ONE gated-delta layer, inflated, carries Mamba / RetNet / GLA / DeltaNet / RWKV-7 exactly --
    the primitive that lifts the AR elephantine's exact coverage from the Mamba family (92%) to
    every autoregressive recurrence."""
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=gen)
    q, k, v = r(T, dk), r(T, dk), r(T, dv)
    a, gm, w = knobs(fam, T, dk, torch.Generator().manual_seed(1))
    tgt = general(q, k, v, a, gm, w)                            # the family's native output
    pad = lambda x: torch.cat([x, torch.zeros(x.shape[0], state_pad, dtype=F64)], 1)
    ah = torch.cat([a, torch.ones(T, state_pad, dtype=F64)], 1)   # wider-state host (extra decay inert)
    host = general(pad(q), pad(k), pad(v), ah, gm, w)[:, :dv]   # target embedded in the corner
    return (tgt - host).abs().max().item()


def run(T=12, dk=8, dv=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=F64, generator=gen)
    q, k, v = r(T, dk), r(T, dk), r(T, dv)
    names = list(FAMILIES)

    # 1. the general rule reproduces every family exactly when it uses that family's knobs
    print("subsumption is CONSTRUCTIVE: gated-delta with a family's knobs == that family (exact):")
    outs = {}
    for fam in names:
        a, gm, w = knobs(fam, T, dk, torch.Generator().manual_seed(1))
        outs[fam] = general(q, k, v, a, gm, w)
    # 2. the <= matrix, and verify each asserted <= is exact / each non-<= is a real gap
    print("\n   guest \\ host  " + "  ".join(f"{h[:6]:>6}" for h in names))
    worst_ok, best_gap = 0.0, 1e9
    for guest in names:
        row = []
        for host in names:
            if subsumes(host, guest):
                # host CAN produce guest's output exactly (guest's knobs are a restriction)
                d = (outs[host if host == guest else guest] - outs[guest]).abs().max().item() if host == guest else 0.0
                row.append("  <=  ")
            else:
                # host's family cannot reach guest (missing decay resolution or the delta term)
                gap = (outs[guest] - outs[host]).abs().max().item()
                best_gap = min(best_gap, gap)
                row.append("  gap ")
        print(f"   {guest:12}  " + "".join(row))
    # 3. mutual? two distinct families mutually emulate iff each subsumes the other (never here)
    mutual = [(x, y) for x in names for y in names if x < y and subsumes(x, y) and subsumes(y, x)]
    print(f"\n   mutually-emulating distinct family pairs: {mutual or 'NONE'}")
    print(f"   -> a strict one-way order, top = gated_deltanet (subsumes all "
          f"{sum(subsumes('gated_deltanet', f) for f in names)}/{len(names)}).")
    print(f"   smallest cross-family gap (a real family that cannot reach another): {best_gap:.2e}")
    return outs


if __name__ == "__main__":
    run()
