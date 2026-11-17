"""A LOOPED host for an UNTIED target under a real pre-norm: the construction and its measured agreement.

The relation lets a looped host (b weight-distinct layers applied K times) host an untied target of
L <= b layers at eff_layers >= L. The host puts the target's layers in its block; iteration 1 must be the
target and iterations 2..K must be the identity. Norm-free, a counter channel does it (`looped_counter_two`);
under a pre-norm a carried counter is read divided by the token's rms, and an attention sublayer cannot be
switched off by a gate shift at all (its output is a convex mix of values, not a unit).

The pre-norm construction (2 extra residual channels: a counter c and a start-token flag f; the input
begins with one fixed token, BOS -- the start-token rule):

  * counter: the block's LAST FFN gets 2d + 2 extra units writing gamma |x|^2 / d (exactly: silu(u) u -
    silu(-u) u = u^2) to c at every token (gamma = 1e-6), and H f_hat^2 at the start token (the flag is
    nonzero only there; H makes c_BOS ~ 1e9 |h_BOS|). In iteration 1 every sublayer reads c = 0: with
    eps' = eps d / (d + 2) and the norm weights rescaled, iteration 1 IS the target (the flag is ~1e-6
    |h_BOS|, a 1e-12 relative perturbation of the start token's norm);
  * FFN off for n >= 2: every target unit's gate pre-activation is shifted by -B c / r (scale-robust: any
    c / r > 0 masks once B gamma / r_max >> 30);
  * attention off for n >= 2: the query and key each get A c_hat in the lowest-frequency RoPE pair, so the
    logit to key j gains ~ A c_hat_i c_hat_j. The start token's c_hat is ~ sqrt(d + 2) (its counter
    dominates its stream), every other token's is ~ gamma n / r: from iteration 2 on every query attends to
    the start token, whose value V x_BOS is ~ |h_BOS| / H ~ 1e-9 of a normal value (its counter has
    swamped its normalised stream). In iteration 1 c = 0, no logit moves;
  * the counter is never erased: gamma K << |h|, so the final norm sees a 1e-12 relative perturbation;
    the START TOKEN's own logits are scaled by ~1 / H (argmax kept, its distribution flattened) -- the one
    position the construction gives up, and the one no continuation is scored on. The flag rides on the
    start token's EMBEDDING, so a later occurrence of the same id (BOS = EOS generated mid-text) gives up
    its own distribution too (kl_ex_start leaves those positions out);
  * a value bias b_v (Qwen2) survives the silenced attention (the start token's value is ~b_v): 4d units
    per layer cancel W_o b_v from iteration 2 on (|x|^2 / d unmasked minus gate-shifted: on iff c > 0);
  * a host that RMS-normalises the stream between iterations (Nanbeige) rescales the start token's counter
    to ~sqrt(d); 2 refresh units per layer, c += G (c / r)^2 (G gamma / r << 1 << G d), keep it swamped;
  * an untied prelude / coda around the looped block: the coda must see the start token's target key and
    value, but its stream is swamped. Every coda K / V projection gets a column on the counter channel
    equal to the start token's target key / value divided by its c_hat there (constants: the start
    token's stream is the same in every input); at other tokens c_hat ~ gamma / r, a 1e-8 perturbation.
    Erasing the counter at the coda instead (host-split-erase) is limited by fp32 cancellation: H ~ 1e4
    leaves ~1e-3 errors either way.

Variants:
  target          the untied target (own forward, checked against transformers)
  host            the construction, block looped K times
  naive           the target's layers looped K times, no counter (what the relation grants for free)
  ffn-only        counter + FFN masks, attention left on in iterations >= 2
  big-carry       as host with gamma = 1 (the norm-free counter's scale)
  all-huge        every token's counter written huge (no start-token trick): argmax kept, logits flattened
  host-internorm  a Nanbeige-style host (RMSNorm between iterations, the target's final weight in the head)
                  with refresh units; -norefresh without them; naive-internorm with no counter
  host-split      untied prelude [0, L/3) + looped middle + untied coda (qwen3loop-style), counter erased
                  at the coda; -noerase without the erase

  python -m hf.looped_host_prenorm HuggingFaceTB/SmolLM2-135M --device cuda:0 --K 2,4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from act.act_swap_agreement import N_NEW, PROMPTS, STORY_PROMPTS

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

GAMMA = 1e-6      # per-iteration counter increment at ordinary tokens
FLAG = 1e-6       # start-token flag, relative to |h_BOS| at the embedding
H_BOS = 1e9       # start-token counter, relative to |h_BOS|
B_FFN = 1e13      # FFN gate shift per unit of c / r (masks any token with r <~ 1e5)
A_ATT = 1e12      # attention marker strength (query side); key side 1

VARIANTS = {   # name: dict(counter, ffn_mask, attn_mask, gamma, bos_huge, all_huge)
    "host": dict(counter=True, ffn=True, attn=True, gamma=GAMMA, all_huge=False),
    "naive": dict(counter=False, ffn=False, attn=False, gamma=GAMMA, all_huge=False),
    "ffn-only": dict(counter=True, ffn=True, attn=False, gamma=GAMMA, all_huge=False),
    "big-carry": dict(counter=True, ffn=True, attn=True, gamma=1.0, all_huge=False),
    "all-huge": dict(counter=True, ffn=True, attn=True, gamma=GAMMA, all_huge=True),
    # Nanbeige-style host: its final RMSNorm also runs between iterations
    "host-internorm": dict(counter=True, ffn=True, attn=True, gamma=GAMMA, inter_norm=True, refresh=1e3),
    "host-internorm-norefresh": dict(counter=True, ffn=True, attn=True, gamma=GAMMA, inter_norm=True),
    "naive-internorm": dict(counter=False, ffn=False, attn=False, gamma=GAMMA, inter_norm=True),
    # qwen3loop-style host: untied prelude [0, L/3), looped middle [L/3, 2L/3), untied coda. The coda must
    # see the start token's target key / value although its stream is swamped: host-split gives every
    # coda K / V projection a column on the counter channel carrying the start token's (constant) target
    # key / value; host-split-erase instead erases the counter at the coda's first FFN (H lowered to 1e4 so
    # the fp32 cancellation stays ~1e-4 |h_BOS|); -noerase does neither
    "host-split": dict(counter=True, ffn=True, attn=True, gamma=GAMMA, split="thirds", coda_cols=True),
    "host-split-erase": dict(counter=True, ffn=True, attn=True, gamma=GAMMA, split="thirds", erase=True, H=1e4),
    "host-split-noerase": dict(counter=True, ffn=True, attn=True, gamma=GAMMA, split="thirds", erase=False, H=1e4),
}


def rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


class Llama:
    """The target's own weights, a pure-torch Llama forward with the host's extra channels."""

    def __init__(self, model):
        self.m = model
        cfg = model.config
        self.cfg = cfg
        self.d = cfg.hidden_size
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or self.d // self.nh
        self.layers = model.model.layers
        self.eps = cfg.rms_norm_eps
        self.mark = self.hd // 2 - 1          # lowest-frequency RoPE pair (j, j + hd/2), rotate_half layout

    @torch.no_grad()
    def forward(self, ids, K=1, v=None, bos_id=None, _bake=False, _record=None):
        """Logits (T, V) of the host (v=None: the plain target). The looped block is layers [a, b) of the
        target (v["split"], default all), run K times; [0, a) is an untied prelude and [b, L) an untied coda.
        v["inter_norm"]: the host applies its final RMSNorm (weight 1, every channel) at the end of every
        iteration, and the target's final norm weight is folded into the head (Nanbeige-style hosts).
        v["erase"]: the first coda FFN removes the start token's counter (a baked constant times f_hat^2)."""
        m, d, dev = self.m, self.d, ids.device
        v = v or {}
        L = len(self.layers)
        a0, b0 = (L // 3, 2 * L // 3) if v.get("split") == "thirds" else v.get("split", (0, L))
        H = v.get("H", H_BOS)
        h = m.model.embed_tokens(ids)[0].float()                         # (T, d)
        T = h.shape[0]
        c = torch.zeros(T, 1, device=dev)
        f = torch.zeros(T, 1, device=dev)
        if v and bos_id is not None:
            f[ids[0] == bos_id] = FLAG * h[0].norm()
        pos = torch.arange(T, device=dev)[None]
        cos, sin = m.model.rotary_emb(h[None], pos)
        cos, sin = cos[0].float(), sin[0].float()
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        bake = {}
        cols = {}
        if v.get("coda_cols") and b0 < L:
            # constants of the construction (the start token's stream is the same in every input): the
            # target's start-token attention input u at each coda layer, and the host's c_hat there
            key = ("cols", K, tuple(sorted((kk, str(vv)) for kk, vv in v.items())))
            store = self.__dict__.setdefault("_baked", {})
            if key not in store:
                rt, rh = {}, {}
                self.forward(ids[:, :1], _record=rt)
                self.forward(ids[:, :1], K=K, v=dict(v, coda_cols=False), bos_id=bos_id, _record=rh)
                cc = {}
                for li in range(b0, L):
                    a = self.layers[li].self_attn
                    u, chb = rt[li][0], rh[li][1]
                    kb = a.k_proj(u) - (a.k_proj.bias if a.k_proj.bias is not None else 0)
                    vb = a.v_proj(u) - (a.v_proj.bias if a.v_proj.bias is not None else 0)
                    cc[li] = (kb.view(self.nkv, self.hd) / chb, vb.view(self.nkv, self.hd) / chb)
                store[key] = cc
            cols = store[key]

        def rms(hh, cc, ff):
            return torch.sqrt((hh.pow(2).sum(-1, keepdim=True) + cc.pow(2) + ff.pow(2)) / d + self.eps)

        def layer_fn(li, h, c, f, looped):
            layer = self.layers[li]
            a = layer.self_attn
            r = rms(h, c, f)
            x = layer.input_layernorm.weight * (h / r)
            if _record is not None:                                   # the start token's attention input, c_hat
                _record[li] = (x[0].clone(), (c / r)[0, 0].item())
            q = a.q_proj(x).view(T, self.nh, self.hd).transpose(0, 1)
            k = a.k_proj(x).view(T, self.nkv, self.hd).transpose(0, 1)
            val = a.v_proj(x).view(T, self.nkv, self.hd).transpose(0, 1)
            if not looped and li in cols:                             # coda: the counter column carries the
                ch = (c / r)[:, 0]                                    # start token's target key / value
                k = k + cols[li][0][:, None, :] * ch[None, :, None]
                val = val + cols[li][1][:, None, :] * ch[None, :, None]
            if looped and v.get("attn"):
                ch = (c / r)[:, 0]
                q, k = q.clone(), k.clone()
                q[:, :, self.mark] += A_ATT * ch
                k[:, :, self.mark] += ch
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            rep = self.nh // self.nkv
            k, val = k.repeat_interleave(rep, 0), val.repeat_interleave(rep, 0)
            s_ = (q @ k.transpose(-1, -2)) / self.hd ** 0.5
            s_ = s_.masked_fill(~causal, float("-inf"))
            o = (F.softmax(s_, -1) @ val).transpose(0, 1).reshape(T, -1)
            h = h + a.o_proj(o)
            mlp = layer.mlp
            r = rms(h, c, f)
            x = layer.post_attention_layernorm.weight * (h / r)
            g = mlp.gate_proj(x)
            if looped and v.get("ffn"):
                g = g - B_FFN * (c / r)
            h_in, c_in = h, c
            h = h + mlp.down_proj(F.silu(g) * mlp.up_proj(x))
            if looped and v.get("attn") and a.v_proj.bias is not None:
                # a value bias b_v survives the silenced attention (the start token's value is ~b_v): cancel
                # W_o b_v in iterations >= 2 with 4d units, |x|^2/d by silu(u) u - silu(-u) u, unmasked
                # minus gate-shifted (on only when c > 0; exactly 0 in iteration 1)
                xu = h_in / r
                sft = B_FFN * (c_in / r)
                q_on = xu.pow(2).sum(-1, keepdim=True) / d
                q_off = ((F.silu(xu - sft) - F.silu(-xu - sft)) * xu).sum(-1, keepdim=True) / d
                bv = a.v_proj.bias.view(self.nkv, self.hd).repeat_interleave(self.nh // self.nkv, 0).reshape(-1)
                h = h - a.o_proj(bv) * (q_on - q_off)
            if looped and v.get("counter") and li == b0 - 1:          # the block's counter units (same FFN)
                xh = h_in / r                                         # the FFN's input, unit weights
                inc = v["gamma"] * xh.pow(2).sum(-1, keepdim=True) / d
                fh = f / r
                if "big" not in bake:                                 # a constant of the construction: the start
                    hb = h_in[0].norm()                               # token's stream is the same in every input
                    bake["big"] = (H * hb / (f[0, 0] / r[0, 0]) ** 2, hb)
                big = bake["big"][0] * fh.pow(2)                      # H |h_BOS| at the start token only
                hb = bake["big"][1]
                if v.get("all_huge"):
                    big = H * hb * xh.pow(2).sum(-1, keepdim=True) / d
                c = c + inc + big
            if looped and v.get("refresh"):                           # 2 units/layer reading the FFN input:
                c = c + v["refresh"] * (c_in / r).pow(2)              # c += G (c/r)^2 keeps the start token swamped
            return h, c

        def norm_all(h, c, f):                                         # weight-1 RMSNorm over every channel
            r = rms(h, c, f)
            return h / r, c / r, f / r

        for li in range(a0):                                           # untied prelude
            h, c = layer_fn(li, h, c, f, False)
        for n in range(K):
            for li in range(a0, b0):
                h, c = layer_fn(li, h, c, f, True)
            if v.get("inter_norm"):
                h, c, f = norm_all(h, c, f)
        if b0 < L and v.get("erase"):                                  # first coda FFN: -c_BOS at the start token
            fh2 = (f / rms(h, c, f)).pow(2)
            if _bake:
                return c[0, 0] / fh2[0, 0]
            key = ("erase", K, tuple(sorted((kk, str(vv)) for kk, vv in v.items())))
            if key not in self.__dict__.setdefault("_baked", {}):
                self._baked[key] = self.forward(ids[:, :1], K=K, v=v, bos_id=bos_id, _bake=True)
            c = c - self._baked[key] * fh2
        for li in range(b0, L):                                        # untied coda
            h, c = layer_fn(li, h, c, f, False)
        if v.get("inter_norm"):
            hn, _, _ = norm_all(h, c, f)
            x = m.model.norm.weight * hn                               # w folded into the head
        else:
            x = m.model.norm.weight * (h / rms(h, c, f))
        return m.lm_head(x)


@torch.no_grad()
def greedy(fn, ids, n_new=N_NEW):
    seq = ids
    for _ in range(n_new):
        seq = torch.cat([seq, fn(seq)[-1:].argmax(-1)[None]], 1)
    return seq[:, ids.shape[1]:]


@torch.no_grad()
def score(fn, enc, ref_seq, ref_lp, ref_cont, start=None):
    """As act/act_swap_agreement.py; kl_ex_start averages KL over the positions whose input token is not
    the start-token id (a recurring start token -- e.g. EOS = BOS generated mid-text -- carries the flag
    too and gives up its own next-token distribution, argmax kept)."""
    top1 = kl = kl_x = 0.0
    n = n_x = ex32 = ex64 = 0
    div = []
    for ids, s, lq, rc in zip(enc, ref_seq, ref_lp, ref_cont):
        P = ids.shape[1]
        lp = F.log_softmax(fn(s)[P - 1:-1].double(), -1).cpu()
        kls = (lq.exp() * (lq - lp)).sum(-1)
        kl += kls.sum().item()
        keep = (s[0, P - 1:-1] != start).cpu() if start is not None else torch.ones_like(kls, dtype=torch.bool)
        kl_x += kls[keep].sum().item()
        n_x += int(keep.sum())
        top1 += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
        n += lp.shape[0]
        g = greedy(fn, ids)
        same = (g[0] == rc[0]).cpu().numpy()
        ex32 += bool(same[:32].all())
        ex64 += bool(same.all())
        div.append(int(np.argmin(same)) if not same.all() else N_NEW)
    return dict(top1=top1 / n, kl=kl / n, kl_ex_start=kl_x / max(n_x, 1), start_positions=n - n_x,
                ex32=ex32 / len(enc), ex64=ex64 / len(enc), div=float(np.mean(div)))


def load(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hf.attn_pin import attn_impl_for
    tok = AutoTokenizer.from_pretrained(model_id)
    for _ in range(30):
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,  # eager (hf/attn_pin.py)
                                                        attn_implementation=attn_impl_for(model_id)).to(device).eval(), tok
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(60)
    raise RuntimeError("GPU busy")


@torch.no_grad()
def run(model_id, device="cuda:0", Ks=(2, 4), story=False, n_prompts=24, variants=None, sink=None):
    model, tok = load(model_id, device)
    net = Llama(model)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    prompts = (STORY_PROMPTS if story else PROMPTS)[:n_prompts]
    enc = []
    for p in prompts:                                     # the start-token rule: every input begins with BOS
        ids = tok(p, return_tensors="pt").input_ids[0]
        if ids[0].item() != bos:
            ids = torch.cat([torch.tensor([bos]), ids])
        enc.append(ids[None].to(device))
    tgt = lambda s: net.forward(s)
    chk = max((tgt(ids) - model(ids).logits[0]).abs().max().item() for ids in enc[:4])
    t0 = time.time()
    ref_cont = [greedy(tgt, ids) for ids in enc]
    ref_seq = [torch.cat([ids, c], 1) for ids, c in zip(enc, ref_cont)]
    ref_lp = [F.log_softmax(tgt(s)[ids.shape[1] - 1:-1].double(), -1).cpu() for s, ids in zip(ref_seq, enc)]
    print(f"{model_id}: L={len(net.layers)} d={net.d}; own forward vs transformers max|d| {chk:.2e}; "
          f"refs in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    rows = []
    for K in Ks:
        for name in variants or VARIANTS:
            v = VARIANTS[name]
            fn = lambda s, v=v, K=K: net.forward(s, K=K, v=v, bos_id=bos)
            r = score(fn, enc, ref_seq, ref_lp, ref_cont, start=bos)
            # the start token's own prediction (not a scored position): does its argmax survive?
            b0 = [(fn(s)[0].argmax() == tgt(s)[0].argmax()).item() for s in ref_seq[:4]]
            row = dict(model=model_id, L=len(net.layers), d=net.d, K=K, variant=name, own_forward_err=chk,
                       bos_argmax_kept=float(np.mean(b0)), n_prompts=len(enc), gamma=v["gamma"], **r)
            rows.append(row)
            sink and sink(row)
            print(f"  K={K} {name:>10}: top1 {r['top1']:.4f} ex32 {r['ex32']:.2f} ex64 {r['ex64']:.2f} "
                  f"KL {r['kl']:.2e} (ex start {r['kl_ex_start']:.2e}, {r['start_positions']} start-id positions) "
                  f"div {r['div']:.1f}", file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", default="2,4")
    ap.add_argument("--story", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--variants")
    ap.add_argument("--out", default="hf/data/looped_host_prenorm.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def sink(r):
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    run(a.model, a.device, [int(k) for k in a.K.split(",")], a.story, a.n_prompts,
        a.variants.split(",") if a.variants else None, sink)


if __name__ == "__main__":
    main()
