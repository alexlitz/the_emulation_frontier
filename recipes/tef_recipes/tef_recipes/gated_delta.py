# Vendored from the_emulation_frontier/attn/gated_delta_layer.py by `python -m recipes.vendor`. Do not edit: edit attn/gated_delta_layer.py.
"""A RUNNABLE gated-delta linear-recurrence layer — the primitive that lifts the
elephantine's *exact* coverage from the Mamba family to EVERY autoregressive recurrence.

`attn/ssm_cross.py` proves the general gated-delta rule subsumes every linear-recurrence
family as *recurrence math* (a bare function over q/k/v tensors). This module makes that
primitive a real `nn.Module`: projections in, the recurrence, projection out — droppable
into an end-to-end language model (`GatedDeltaLM`). Two facts are then verifiable as a
MODEL forward, not just a scan over hand-made tensors:

  1. `GatedDeltaLayer(family="gated_deltanet")` reproduces transformers' own shipped
     `torch_recurrent_gated_delta_rule` (the Qwen3-Next / gated-DeltaNet kernel) token-for-
     token — so the layer is the *real library* recurrence, not a look-alike.
  2. `inflate` embeds a NARROWER guest layer into a WIDER host (more state per head) so the
     host's forward equals the guest's exactly — the state-inflation carry, run as a module.
     `GatedDeltaLM` + `inflate_lm` do the same for a whole model: a wider host LM runs a
     smaller guest LM's compressed recurrence token-for-token.

Every non-Mamba linear-recurrence in the zoo (gated-DeltaNet in Qwen3-Next, RWKV-7,
MiniMax-lightning, LFM2, RG-LRU) is a knob-restriction of this layer's general rule
(a per-key-channel decay, a delta-remove strength, a write strength — see `FAMILIES`), so a
single wider gated-delta host carries any of them. Together with the softmax attention host
(all attention/diffusion) and the Mamba-SSM host (the Mamba family), this closes the exact
sub-zoo to every recurrence in the zoo.

Run:  python -m attn.gated_delta_layer
"""
from __future__ import annotations

import torch
from torch import nn

F64 = torch.float64

# how each family sets the general rule's knobs (mirrors attn/ssm_cross.FAMILIES):
#   decay: 0 const-1 | 1 const-scalar | 2 data scalar_t | 3 data per-channel ;  delta: 0/1
FAMILIES = {
    "linear":         dict(decay=0, delta=0),
    "retnet":         dict(decay=1, delta=0),
    "mamba2":         dict(decay=2, delta=0),
    "gla":            dict(decay=3, delta=0),   # == Mamba-1 (diagonal selective SSM)
    "deltanet":       dict(decay=0, delta=1),
    "gated_deltanet": dict(decay=3, delta=1),   # RWKV-7 / RG-LRU / lightning / LFM2 live here
}


def gd_recurrence(q, k, v, a, gamma, w):
    """Batched general gated-delta rule (the `attn.ssm_cross.general` recurrence over B,H):
        S_t = Diag(a_t) S_{t-1} - gamma_t (S_{t-1} k_t) k_t^T + w_t v_t k_t^T,  y_t = S_t q_t
    q,k,a:(B,H,T,dk)  v:(B,H,T,dv)  gamma,w:(B,H,T)  ->  y:(B,H,T,dv)."""
    B, Hn, T, dk = k.shape
    dv = v.shape[-1]
    S = torch.zeros(B, Hn, dv, dk, dtype=q.dtype)
    out = []
    for t in range(T):
        S = S * a[:, :, t, None, :]                                    # per-key-channel decay
        Sk = (S * k[:, :, t, None, :]).sum(-1)                         # S @ k_t  -> (B,H,dv)
        S = S - gamma[:, :, t, None, None] * Sk[..., None] * k[:, :, t, None, :]   # delta remove
        S = S + w[:, :, t, None, None] * v[:, :, t, :, None] * k[:, :, t, None, :]  # write v ⊗ k
        out.append((S * q[:, :, t, None, :]).sum(-1))                  # S @ q_t
    return torch.stack(out, 2)


class GatedDeltaLayer(nn.Module):
    """A runnable multi-head gated-delta mixer. `family` picks how the decay / delta knobs
    are produced (a knob-restriction of the general rule); the recurrence and the q/k/v/o
    projections are shared. Runs in fp64 by default so the carry is exact."""

    def __init__(self, d_model, n_heads, head_dim, family="gated_deltanet", dtype=F64):
        super().__init__()
        self.n_heads, self.head_dim, self.family = n_heads, head_dim, family
        self.decay, self.delta = FAMILIES[family]["decay"], FAMILIES[family]["delta"]
        H, hd, inner = n_heads, head_dim, n_heads * head_dim
        lin = lambda a, b, bias=False: nn.Linear(a, b, bias=bias).to(dtype)
        self.q_proj, self.k_proj, self.v_proj = lin(d_model, inner), lin(d_model, inner), lin(d_model, inner)
        self.o_proj = lin(inner, d_model)
        if self.decay == 1:                                           # one learned scalar decay
            self.g_scalar = nn.Parameter(torch.zeros((), dtype=dtype))
        elif self.decay == 2:                                         # data-dependent scalar per head
            self.g_proj = lin(d_model, H, bias=True)
        elif self.decay == 3:                                         # data-dependent per key-channel
            self.g_proj = lin(d_model, inner, bias=True)
        if self.delta:                                               # beta gates delta-remove == write
            self.b_proj = lin(d_model, H, bias=True)

    def knobs(self, x):
        """Compute per-head (a, gamma, w) tensors from x — the family's knob-restriction."""
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        if self.decay == 0:
            a = torch.ones(B, H, T, hd, dtype=x.dtype)
        elif self.decay == 1:
            a = torch.sigmoid(self.g_scalar).expand(B, H, T, hd)
        elif self.decay == 2:
            a = torch.sigmoid(self.g_proj(x)).transpose(1, 2)[..., None].expand(B, H, T, hd)
        else:
            a = torch.sigmoid(self.g_proj(x)).view(B, T, H, hd).transpose(1, 2)
        if self.delta:
            beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)      # (B,H,T)
            gamma, w = beta, beta                                     # gated-DeltaNet ties them
        else:
            gamma = torch.zeros(B, H, T, dtype=x.dtype)
            w = torch.ones(B, H, T, dtype=x.dtype)
        return a, gamma, w

    def forward(self, x):
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        shp = lambda p: p(x).view(B, T, H, hd).transpose(1, 2)        # (B,H,T,hd)
        q, k, v = shp(self.q_proj), shp(self.k_proj), shp(self.v_proj)
        a, gamma, w = self.knobs(x)
        y = gd_recurrence(q, k, v, a, gamma, w)                       # (B,H,T,hd)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, H * hd))


@torch.no_grad()
def inflate(guest: GatedDeltaLayer, head_dim: int) -> GatedDeltaLayer:
    """State-inflation carry, as a module: build a host with a WIDER per-head state
    (head_dim >= guest.head_dim) whose forward equals the guest's exactly. The guest's
    q/k/v occupy the first `guest.head_dim` channels of each head; the padded key-channels
    get zero keys (and zero decay input) so their state stays 0 and the extra value-channels
    never reach the output (o_proj zeroes their columns)."""
    g, H, hg, hh = guest, guest.n_heads, guest.head_dim, head_dim
    assert hh >= hg, "host per-head state must be at least the guest's"
    d_model = g.q_proj.in_features
    host = GatedDeltaLayer(d_model, H, hh, g.family, dtype=g.q_proj.weight.dtype)

    def embed_rows(dst, src):                                        # per-head: first hg rows := src, rest 0
        dst.weight.zero_()
        dst.weight.view(H, hh, -1)[:, :hg].copy_(src.weight.view(H, hg, -1))
        if dst.bias is not None and src.bias is not None:
            dst.bias.zero_(); dst.bias.view(H, hh)[:, :hg].copy_(src.bias.view(H, hg))
    for name in ("q_proj", "k_proj", "v_proj"):
        embed_rows(getattr(host, name), getattr(g, name))
    if g.decay == 1:                                                 # scalar decay: same param (padding inert)
        host.g_scalar.copy_(g.g_scalar)
    elif g.decay == 2:                                              # scalar-per-head proj: unchanged
        host.g_proj.load_state_dict(g.g_proj.state_dict())
    elif g.decay == 3:                                              # per-channel decay: embed, pad inert
        embed_rows(host.g_proj, g.g_proj)
    if g.delta:
        host.b_proj.load_state_dict(g.b_proj.state_dict())
    host.o_proj.weight.zero_()                                       # per-head: first hg cols := guest, rest 0
    host.o_proj.weight.view(d_model, H, hh)[:, :, :hg].copy_(g.o_proj.weight.view(d_model, H, hg))
    return host


class _RMSNorm(nn.Module):
    def __init__(self, d, dtype=F64, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d, dtype=dtype)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class _SwiGLU(nn.Module):
    def __init__(self, d, d_ff, dtype=F64):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False).to(dtype)
        self.up = nn.Linear(d, d_ff, bias=False).to(dtype)
        self.down = nn.Linear(d_ff, d, bias=False).to(dtype)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class GatedDeltaLM(nn.Module):
    """A tiny end-to-end LM whose sequence mixer is the gated-delta layer: embed ->
    [RMSNorm + GatedDelta + res, RMSNorm + SwiGLU + res] x L -> RMSNorm -> lm_head."""

    def __init__(self, vocab, d_model, n_layers, n_heads, head_dim, family="gated_deltanet",
                 d_ff=None, dtype=F64):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.embed = nn.Embedding(vocab, d_model).to(dtype)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict(dict(
                n1=_RMSNorm(d_model, dtype), mix=GatedDeltaLayer(d_model, n_heads, head_dim, family, dtype),
                n2=_RMSNorm(d_model, dtype), mlp=_SwiGLU(d_model, d_ff, dtype))))
        self.norm = _RMSNorm(d_model, dtype)
        self.lm_head = nn.Linear(d_model, vocab, bias=False).to(dtype)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = x + b["mix"](b["n1"](x))
            x = x + b["mlp"](b["n2"](x))
        return self.lm_head(self.norm(x))


@torch.no_grad()
def inflate_lm(guest: GatedDeltaLM, head_dim: int, d_ff: int) -> GatedDeltaLM:
    """A WIDER host LM (bigger per-head state + wider FFN, same d_model) that runs the guest
    LM's compressed recurrence token-for-token. Mixer carried by `inflate` (state-inflation);
    FFN zero-padded to the wider width (SiLU(0)*0 = 0, so the extra units are inert)."""
    g = guest
    b0 = g.blocks[0]
    host = GatedDeltaLM(g.embed.num_embeddings, g.embed.embedding_dim, len(g.blocks),
                        b0["mix"].n_heads, head_dim, b0["mix"].family, d_ff,
                        dtype=g.embed.weight.dtype)
    host.embed.load_state_dict(g.embed.state_dict())
    host.norm.load_state_dict(g.norm.state_dict())
    host.lm_head.load_state_dict(g.lm_head.state_dict())
    for hb, gb in zip(host.blocks, g.blocks):
        hb["n1"].load_state_dict(gb["n1"].state_dict())
        hb["n2"].load_state_dict(gb["n2"].state_dict())
        hb["mix"].load_state_dict(inflate(gb["mix"], head_dim).state_dict())
        gf, it = gb["mlp"], gb["mlp"].gate.out_features                # zero-pad SwiGLU to wider d_ff
        for name in ("gate", "up"):
            getattr(hb["mlp"], name).weight.zero_()
            getattr(hb["mlp"], name).weight[:it].copy_(getattr(gf, name).weight)
        hb["mlp"].down.weight.zero_(); hb["mlp"].down.weight[:, :it].copy_(gf.down.weight)
    return host


def _demo():
    torch.manual_seed(0)
    print("gated-delta layer: a wider host carries each family token-for-token (module forward)")
    x = torch.randn(2, 9, 16, dtype=F64)
    for fam in FAMILIES:
        guest = GatedDeltaLayer(16, 4, 6, fam).eval()
        host = inflate(guest, head_dim=10).eval()
        d = (guest(x) - host(x)).abs().max().item()
        print(f"  {fam:16} host(6->10)  max|Δ| = {d:.1e}")
    ids = torch.randint(0, 40, (2, 9))
    guest = GatedDeltaLM(40, 16, 3, 4, 6, "gated_deltanet", d_ff=32).eval()
    host = inflate_lm(guest, head_dim=10, d_ff=48).eval()
    d = (guest(ids) - host(ids)).abs().max().item()
    print(f"\n  end-to-end LM: wider host (state 6->10, ffn 32->48) max|Δ logits| = {d:.1e}")


if __name__ == "__main__":
    _demo()
