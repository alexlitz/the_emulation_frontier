"""The exact pre-norm MoE select on a strong router margin (Lean EP.moe_select_prenorm_exact, EP.pairSel_sq_sum,
built on EP.const_norm_carry).

Target: a pre-norm (RMSNorm eps > 0, weight w) top-1, unit-gate MoE with ReLU experts. Router logits
z = Wr N(x), expert pre-activations p_ej = P_ej . N(x), output sum_j relu(p_e*j) v_e*j.

Host: residual [x; g; kappa] (E^2 select channels g, one pad kappa), ONE RMSNorm over all of it with
eps' = (d eps - S)/(d + E^2 + 1), S = E(E-1)/2 delta^2, weight [sqrt(d/D) w; 1], and two ReLU FFN layers with
biases (moe/dense_host_prenorm builds the LIMIT variant instead: SwiGLU select, carry at gamma = 1e-8):

* select layer, reading [x; 0; sqrt S]: the clamped pair features g_ee' = clamp_delta(Lam (z_e' - z_e)),
  clamp_delta(t) = relu(t) - relu(t - delta) (2E^2 ReLU units, a bias each), and -sqrt S on the pad (a bias);
* expert layer, reading [x; g; 0]: expert e's units masked by lam sum_e' (normed g)_ee', the 2E^2 clamp units
  recomputed from the same exact logits to erase g, and +sqrt S back on the pad.

On a strong margin (every pairwise gap Lam |z_e - z_e'| >= delta) the select vector takes only the values 0 and
delta, with squared norm S at both reads, so both norm reads give N(x) exactly. The masks are 0 for the winner
and >= delta/R for the losers; with lam delta / R above every loser pre-activation the expert layer is the
winner's FFN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from norm.const_norm import const_norm_host


def clamp_u(delta, t):
    """EP.clampU: relu(t) - relu(t - delta)."""
    return torch.relu(t) - torch.relu(t - delta)


def pair_sel(z, delta, Lam):
    """EP.pairSel: g[.., e, e'] = clamp_delta(Lam (z_e' - z_e))."""
    return clamp_u(delta, Lam * (z[..., None, :] - z[..., :, None]))


def strong_margin(z, delta, Lam):
    """EP.StrongMargin: every pairwise gap Lam |z_e - z_e'| >= delta (per token)."""
    E = z.shape[-1]
    gap = (Lam * (z[..., :, None] - z[..., None, :])).abs()
    return ((gap >= delta) | torch.eye(E, dtype=torch.bool)).flatten(-2).all(-1)


def rms_norm(h, w, eps):
    return w * h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)


def target_moe(x, eps, w, Wr, P, V):
    """The target's top-1 ReLU MoE on the normed input (unit gate). x (.., d); Wr (E, d); P (E, ew, d);
    V (E, ew, dout). Returns (output, z, p)."""
    N = rms_norm(x, w, eps)
    z = N @ Wr.T
    p = torch.einsum("...d,ejd->...ej", N, P)
    e = z.argmax(-1)
    act = torch.relu(p)
    out = torch.einsum("...ej,ejo->...eo", act, V)
    return out.gather(-2, e[..., None, None].expand(*e.shape, 1, out.shape[-1])).squeeze(-2), z, p


@dataclass
class ReLUFFN:
    """r -> W_out relu(W_in n + b_in) + b_out."""
    W_in: torch.Tensor
    b_in: torch.Tensor
    W_out: torch.Tensor
    b_out: torch.Tensor

    def __call__(self, n):
        return torch.relu(n @ self.W_in.T + self.b_in) @ self.W_out.T + self.b_out


@dataclass
class ExactSelectHost:
    d: int
    E: int
    S: float
    norm_w: torch.Tensor
    eps2: float
    select: ReLUFFN
    expert: ReLUFFN

    @property
    def D(self):
        return self.d + self.E * self.E + 1

    def embed(self, x):
        """[x; 0; sqrt S]."""
        pad = torch.zeros(*x.shape[:-1], self.E * self.E + 1, dtype=x.dtype)
        pad[..., -1] = math.sqrt(self.S)
        return torch.cat([x, pad], -1)

    def norm(self, r):
        return rms_norm(r, self.norm_w, self.eps2)

    def __call__(self, x, trace=None):
        r0 = self.embed(x)
        n0 = self.norm(r0)
        r1 = r0 + self.select(n0)
        n1 = self.norm(r1)
        r2 = r1 + self.expert(n1)
        if trace is not None:
            trace.update(r0=r0, n0=n0, r1=r1, n1=n1, r2=r2)
        return r2


def _clamp_units(Wr, delta, Lam, d, D, dt):
    """2E^2 units (relu(t), relu(t - delta)) per ordered pair, t = Lam (z_e' - z_e), reading the stream."""
    E = Wr.shape[0]
    rows = Lam * (Wr[None, :, :] - Wr[:, None, :])               # [e, e'] -> Lam (Wr[e'] - Wr[e])
    W = torch.zeros(2 * E * E, D, dtype=dt)
    W[0::2, :d] = rows.reshape(E * E, -1)
    W[1::2, :d] = rows.reshape(E * E, -1)
    b = torch.zeros(2 * E * E, dtype=dt)
    b[1::2] = -delta
    return W, b


def build_exact_select_host(eps, w, Wr, P, V, delta, Lam, lam):
    """EP.moe_select_prenorm_exact's host for the target (eps, w, Wr, P, V). V (E, ew, d) writes the stream."""
    E, d = Wr.shape
    ew = P.shape[1]
    dt = Wr.dtype
    D = d + E * E + 1
    S = E * (E - 1) / 2 * delta ** 2
    norm_w, eps2 = const_norm_host(w, eps, torch.ones(E * E + 1, dtype=dt), S)
    gch = d + torch.arange(E * E)                                 # channel of pair (e, e') = d + e E + e'
    # select layer
    Wc, bc = _clamp_units(Wr, delta, Lam, d, D, dt)
    Wo = torch.zeros(D, 2 * E * E, dtype=dt)
    Wo[gch, 2 * torch.arange(E * E)] = 1.0
    Wo[gch, 2 * torch.arange(E * E) + 1] = -1.0
    bo = torch.zeros(D, dtype=dt)
    bo[-1] = -math.sqrt(S)
    select = ReLUFFN(Wc, bc, Wo, bo)
    # expert layer: masked expert units, then the erase units
    We = torch.zeros(E * ew, D, dtype=dt)
    We[:, :d] = P.reshape(E * ew, d)
    for e in range(E):
        We[e * ew:(e + 1) * ew, d + e * E: d + (e + 1) * E] = -lam
    Wout_e = torch.zeros(D, E * ew, dtype=dt)
    Wout_e[:d] = V.reshape(E * ew, d).T
    Wout_c = torch.zeros(D, 2 * E * E, dtype=dt)
    Wout_c[gch, 2 * torch.arange(E * E)] = -1.0
    Wout_c[gch, 2 * torch.arange(E * E) + 1] = 1.0
    bo2 = torch.zeros(D, dtype=dt)
    bo2[-1] = math.sqrt(S)
    expert = ReLUFFN(torch.cat([We, Wc]), torch.cat([torch.zeros(E * ew, dtype=dt), bc]),
                     torch.cat([Wout_e, Wout_c], 1), bo2)
    return ExactSelectHost(d, E, S, norm_w, eps2, select, expert)
