"""Whole-model hosts for the limit families of Lean LimitInstances2.lean, on tiny random models.

Each family replaces one feature in EVERY layer of an n-layer target stack and is a limit in one parameter:

  mask        hard mask S  <-  full attention with -B outside S        B -> +inf   (EP.mask_*)
  sink        plain        <-  an extra key with logit b and value 0   b -> -inf   (EP.sink_*)
  cap         plain        <-  attention-softcapped logits c tanh(s/c) c -> +inf   (EP.cap_*)
  router      top-1 MoE    <-  dense soft router softmax(beta z)        beta -> inf (EP.soft_router_*)
  scaled      read r(x)    <-  the same read times lambda(x), lambda the padded host norm's factor with a
                               carried vector delta s on the pad (EP.vanishing_carry_rate)  delta -> 0
                               (EP.scaled_read_*)
  compensator read u(x)    <-  u(x) t/kappa (a pre-norm host's read, t the normed constant pad) times a
                               fitted g(t) ~ kappa/t                    fit error rho -> 0 (EP.compensator_*)

The attention stacks follow EP.attnLayer: one head per layer, x_t + attn(S_t, sc(x)_t, val(x)), the output
projection folded into val(y) = Mb tanh(Wv y) (so |val| <= Mb, the theorems' value bound). The layer pieces are
the repo's own: hf/local_full.band_mask for the mask (clamped at -B it is EP.maskHostLogits),
attn/attention_sink.sink_attention for the sink, attn/softcap.softcap for the cap,
moe/dense_emulates_moe.moe_top1 / dense_softrouter for the router layers.

States are (T, d) tensors (or (d,) for the per-token stacks) under the sup metric, the head is the last
position's logits W_h x_{T-1} under the sup metric (Lh = ||W_h||_inf->inf), so every distance in the theorems is
a max-abs. `lipschitz` MEASURES a layer's constant: the largest inf->inf norm of its Jacobian over the given
states and random points near them (clamped below at 1, as the theorems ask). It is a sample maximum, not a
certified Lipschitz constant.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from attn.attention_sink import sink_attention
from attn.softcap import softcap
from hf.local_full import band_mask
from moe import dense_emulates_moe as dem

F64 = torch.float64


# --- measured constants ---------------------------------------------------------------------------------

def opnorm_inf(J):
    """inf->inf operator norm: the max row abs sum."""
    return J.abs().sum(-1).max().item()


def lipschitz(f, points, n_perturb=2, radius=0.05, seed=0):
    """max(1, max_p ||Df(p)||_inf->inf) over `points` and n_perturb random points within `radius` (sup) of each."""
    g = torch.Generator().manual_seed(seed)
    best = 1.0
    for p in points:
        cands = [p] + [p + radius * (2 * torch.rand(p.shape, dtype=p.dtype, generator=g) - 1)
                       for _ in range(n_perturb)]
        for q in cands:
            J = torch.autograd.functional.jacobian(lambda z: f(z.view(p.shape)).reshape(-1), q.reshape(-1))
            best = max(best, opnorm_inf(J))
    return best


def run(layers, x):
    """All states x_0 .. x_n of the stack."""
    xs = [x]
    for f in layers:
        xs.append(f(xs[-1]))
    return xs


def model_bound(Lh, Ls, per_layer):
    """EP.model_error_bound_head's form with a uniform per-layer error: Lh (prod L_i) n eps."""
    return Lh * math.prod(Ls) * len(Ls) * per_layer


# --- attention stacks (mask / sink / cap) ---------------------------------------------------------------

class AttnStack:
    """n one-head attention layers of width d over T positions, keys S_t (bool (T, T)), val = Mb tanh(Wv .)."""

    def __init__(self, n, T, d, dh=4, Mb=1.0, W=None, qk_scale=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n, self.T, self.d, self.Mb = n, T, d, Mb
        r = lambda *s: torch.randn(*s, dtype=F64, generator=g)          # noqa: E731
        self.Wq = [r(dh, d) * qk_scale / math.sqrt(d) for _ in range(n)]
        self.Wk = [r(dh, d) * qk_scale / math.sqrt(d) for _ in range(n)]
        self.Wv = [r(d, d) / math.sqrt(d) for _ in range(n)]
        self.band = band_mask(T, W if W is not None else T)[0, 0]          # 0 inside S, finfo.min outside
        self.S = self.band == 0

    def scores(self, i, x):
        return (x @ self.Wq[i].T) @ (x @ self.Wk[i].T).T

    def val(self, i, x):
        return self.Mb * torch.tanh(x @ self.Wv[i].T)

    def target_layer(self, i):
        def f(x):
            s = self.scores(i, x) + self.band
            return x + torch.softmax(s, -1) @ self.val(i, x)
        return f

    def mask_host_layer(self, i, B):
        """EP.maskHostLogits over ALL T keys: band_mask clamped at -B."""
        def f(x):
            s = self.scores(i, x) + self.band.clamp(min=-B)
            return x + torch.softmax(s, -1) @ self.val(i, x)
        return f

    def sink_host_layer(self, i, b):
        """EP.sinkLayer: an extra key with logit b and value 0 (attn/attention_sink.sink_attention per query)."""
        def f(x):
            s, v = self.scores(i, x), self.val(i, x)
            rows = [sink_attention(s[t][self.S[t]], v[self.S[t]], b) for t in range(self.T)]
            return x + torch.stack(rows)
        return f

    def cap_host_layer(self, i, c):
        """EP.capLogits: c tanh(s / c) on the target's keys (attn/softcap.softcap)."""
        def f(x):
            s = softcap(self.scores(i, x), c) + self.band
            return x + torch.softmax(s, -1) @ self.val(i, x)
        return f

    def layers(self, kind, p=None):
        mk = dict(target=lambda i: self.target_layer(i), mask=lambda i: self.mask_host_layer(i, p),
                  sink=lambda i: self.sink_host_layer(i, p), cap=lambda i: self.cap_host_layer(i, p))[kind]
        return [mk(i) for i in range(self.n)]

    def logit_max(self, states):
        """R of EP.logitBox: the largest |sc_i(x_i) t j| over all pairs, x_i the input of layer i."""
        return max(self.scores(i, x).abs().max().item() for i, x in enumerate(states[:-1]))


# --- the soft router stack (per token) ------------------------------------------------------------------

class RouterStack:
    """n layers x + Y_{e*}(x) (moe/dense_emulates_moe.moe_top1), host x + sum_e softmax(beta z)_e Y_e(x)
    (dense_softrouter). Experts silu(x W1_e) W2_e, router z = x Wg."""

    def __init__(self, n, d, E, h=6, seed=0, scale=0.5):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, dtype=F64, generator=g)          # noqa: E731
        self.n, self.d, self.E = n, d, E
        self.W1 = [r(E, d, h) * scale / math.sqrt(d) for _ in range(n)]
        self.W2 = [r(E, h, d) * scale / math.sqrt(h) for _ in range(n)]
        self.Wg = [r(d, E) for _ in range(n)]

    def target_layer(self, i):
        return lambda x: x + dem.moe_top1(x[None], self.W1[i], self.W2[i], self.Wg[i])[0]

    def host_layer(self, i, beta):
        return lambda x: x + dem.dense_softrouter(x[None], self.W1[i], self.W2[i], self.Wg[i], beta)[0]

    def layers(self, kind, beta=None):
        return [self.target_layer(i) if kind == "target" else self.host_layer(i, beta) for i in range(self.n)]

    def margin_min(self, states):
        """delta_r: the smallest router margin z_(1) - z_(2) over the inputs of every layer."""
        out = math.inf
        for i, x in enumerate(states[:-1]):
            z = (x @ self.Wg[i]).sort(descending=True).values
            out = min(out, (z[0] - z[1]).item())
        return out

    def expert_max(self, states):
        """Mb: max_e ||Y_e(x)||_inf over the inputs of every layer."""
        return max(dem.experts_all(x[None], self.W1[i], self.W2[i]).abs().max().item()
                   for i, x in enumerate(states[:-1]))


# --- relative-scale reads (scaled read / compensator), per token ----------------------------------------

def rms_norm(x, w, eps):
    return w * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class ReadStack:
    """n layers G_i(x, r_i(x)) = x + A_i r_i(x), r_i(x) = RMSNorm_eps(x) w_i (the target's pre-norm read).

    scaled host: the read through the padded host norm over [x; delta s_i] (eps' = eps d/(d+e), weight
    [sqrt(d/(d+e)) w_i; 1]): lambda(x) r_i(x) with lambda = rms_t / rms_host (EP.vanishing_carry_rate)."""

    def __init__(self, n, d, e=2, eps=0.05, seed=0):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, dtype=F64, generator=g)          # noqa: E731
        self.n, self.d, self.e, self.eps = n, d, e, eps
        self.A = [r(d, d) * 0.4 / math.sqrt(d) for _ in range(n)]
        self.w = [1 + 0.3 * r(d) for _ in range(n)]
        self.s = [r(e) for _ in range(n)]

    def read(self, i, x):
        return rms_norm(x, self.w[i], self.eps)

    def target_layer(self, i):
        return lambda x: x + self.A[i] @ self.read(i, x)

    def host_read(self, i, x, delta):
        d, e = self.d, self.e
        W = torch.cat([math.sqrt(d / (d + e)) * self.w[i], torch.ones(e, dtype=F64)])
        h = torch.nn.functional.rms_norm(torch.cat([x, delta * self.s[i]]), (d + e,), W, self.eps * d / (d + e))
        return h[:d]

    def host_layer(self, i, delta):
        return lambda x: x + self.A[i] @ self.host_read(i, x, delta)

    def layers(self, kind, delta=None):
        return [self.target_layer(i) if kind == "target" else self.host_layer(i, delta) for i in range(self.n)]

    def rho(self, delta):
        """EP.vanishing_carry_rate's |lambda - 1| <= delta^2 ||s||^2 / (2 d eps), the max over layers."""
        return max(delta ** 2 * s.pow(2).sum().item() / (2 * self.d * self.eps) for s in self.s)

    def K(self):
        return max(opnorm_inf(A) for A in self.A)

    def read_max(self):
        """|r_i(x)|_inf <= max|w_i| sqrt(d) for every x (an RMSNorm output's entries are <= sqrt(d) in size)."""
        return max(w.abs().max().item() for w in self.w) * math.sqrt(self.d)


class CompensatorStack:
    """n layers A_i(x) + u_i(x), A_i(x) = x + V_i tanh(P_i x), u_i(x) = U_i x (a read whose magnitude matters).

    host: a pre-norm host sees u through its norm over [x; kappa] (weight 1, eps'): U_i x / rms_host(x) =
    u_i(x) t(x) / kappa, t(x) = kappa / rms_host(x) the normed pad, and multiplies by a fitted g_i(t) ~ kappa/t
    (EP.compFactor). `fit(deg, tmin, tmax)` fits g by least squares on [tmin, tmax]; rho is its sup relative
    error |g(tau) tau / kappa - 1| on a fine grid."""

    def __init__(self, n, d, kappa=1.0, eps=1e-3, seed=0):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, dtype=F64, generator=g)          # noqa: E731
        self.n, self.d, self.kappa, self.eps = n, d, kappa, eps
        self.V = [r(d, d) * 0.3 / math.sqrt(d) for _ in range(n)]
        self.P = [r(d, d) / math.sqrt(d) for _ in range(n)]
        self.U = [r(d, d) * 0.3 / math.sqrt(d) for _ in range(n)]
        self.coef = None

    def A(self, i, x):
        return x + self.V[i] @ torch.tanh(self.P[i] @ x)

    def u(self, i, x):
        return self.U[i] @ x

    def t(self, x):
        h = torch.cat([x, torch.tensor([self.kappa], dtype=F64)])
        return (self.kappa / torch.sqrt(h.pow(2).mean() + self.eps))

    def fit(self, deg, tmin, tmax):
        tau = np.linspace(tmin, tmax, 2001)
        self.coef = np.polynomial.polynomial.polyfit(tau, self.kappa / tau, deg)
        grid = np.linspace(tmin, tmax, 20001)
        self.rho = float(np.abs(np.polynomial.polynomial.polyval(grid, self.coef) * grid / self.kappa - 1).max())
        self.trange = (tmin, tmax)
        return self.rho

    def g(self, t):
        c = torch.tensor(self.coef, dtype=F64)
        return sum(c[k] * t ** k for k in range(len(c)))

    def target_layer(self, i):
        return lambda x: self.A(i, x) + self.u(i, x)

    def host_layer(self, i):
        def f(x):
            t = self.t(x)
            return self.A(i, x) + (self.g(t) * t / self.kappa) * self.u(i, x)
        return f

    def layers(self, kind):
        return [self.target_layer(i) if kind == "target" else self.host_layer(i) for i in range(self.n)]


# --- greedy generation ----------------------------------------------------------------------------------

class SeqEmbed:
    """EP's embed : List Tok -> Fin T -> V: the last T tokens (left-padded with token 0), E[tok] + P[pos]."""

    def __init__(self, vocab, T, d, seed=0, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(vocab, d, dtype=F64, generator=g) * scale
        self.P = torch.randn(T, d, dtype=F64, generator=g) * scale * 0.5
        self.T = T

    def __call__(self, seq):
        s = ([0] * self.T + list(seq))[-self.T:]
        return self.E[s] + self.P


class BagEmbed:
    """embed : List Tok -> V for the per-token stacks: a decaying bag of the last tokens."""

    def __init__(self, vocab, d, seed=0, decay=0.6, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(vocab, d, dtype=F64, generator=g) * scale
        self.decay = decay

    def __call__(self, seq):
        out = torch.zeros(self.E.shape[1], dtype=F64)
        for k, tok in enumerate(reversed(list(seq)[-6:])):
            out = out + self.decay ** k * self.E[tok]
        return out


def greedy(layers, head, embed, prompt, k):
    """EP.greedy with argmax picks: (tokens, per-step state trajectories, per-step logits)."""
    seq, trajs, logits = list(prompt), [], []
    for _ in range(k):
        xs = run(layers, embed(seq))
        z = head(xs[-1])
        trajs.append(xs)
        logits.append(z)
        seq.append(int(z.argmax()))
    return seq, trajs, logits


def margin(z):
    """EP.MarginAt's delta at the pick: top1 - top2."""
    s = z.sort(descending=True).values
    return (s[0] - s[1]).item()
